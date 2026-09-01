from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
import sys
from typing import Any
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from app.main import app

from app.planning.ai_analytical_planner import (
    AIPlannerProposal,
    RawAIPlannerOutput,
)

from app.ai.native_tool_calling import (
    NativeToolCallAttempt,
    NativeToolCallProposal,
    NativeToolCallRequestResult,
    expected_tool_arguments,
    native_tool_spec_for_contract,
)


# ============================================================
# PURPOSE
# ============================================================
#
# Deterministic E2E regression for the real prompt endpoint:
#
#   POST /planning/ai-native-run
#
# It keeps the production path:
#
#   HTTP endpoint
#   -> server-owned handoff boundary
#   -> analytical views
#   -> planner catalog
#   -> intent routing
#   -> real AI planner Python validation / normalization
#   -> native tool validation
#   -> deterministic executor
#   -> chart payload
#
# It removes only external/non-deterministic dependencies:
#
#   - no real Ollama / Gemma inference;
#   - no real Ollama / Qwen inference;
#   - Gemma's structured wire response is replaced at the
#     narrow inference boundary, while the real Python planner
#     validation / normalization remains active;
#   - no persistent artifact-store write;
#   - no observability-store write;
#   - no pre-existing Preparation SQLite workflow required.
#
# This makes the suite suitable as a fast CI regression for the
# nine currently validated prompt-analysis cases.
# ============================================================


WORKFLOW_ID = "prep:c1a1a1e2e00000000000000000000001"
PLANNER_MODEL = "gemma3:4b"
TOOL_MODEL = "qwen2.5:1.5b-instruct"


@dataclass(frozen=True)
class Case:
    name: str
    objective: str
    family: str
    tool: str
    proposal: dict[str, Any]
    bindings: dict[str, str]
    chart_types: set[str]
    result_statuses: set[str] = field(
        default_factory=lambda: {"complete"}
    )
    derived_token: str | None = None
    source_dataset: bool = False


CASES: tuple[Case, ...] = (
    Case(
        name="CA par catégorie",
        objective="CA par catégorie",
        family="aggregation",
        tool="run_aggregation",
        proposal={
            "analytical_grain": "category",
            "group_column": "category",
            "value_column": "sum_gross_amount",
            "aggregation_function": "sum",
        },
        bindings={
            "group": "category",
            "value": "sum_gross_amount",
        },
        chart_types={"bar"},
        derived_token=":category:category:gross_amount",
    ),
    Case(
        name="CA par pays",
        objective="CA par pays",
        family="aggregation",
        tool="run_aggregation",
        proposal={
            "analytical_grain": "country",
            "group_column": "country",
            "value_column": "sum_gross_amount",
            "aggregation_function": "sum",
        },
        bindings={
            "group": "country",
            "value": "sum_gross_amount",
        },
        chart_types={"bar"},
        derived_token=":category:country:gross_amount",
    ),
    Case(
        name="CA par marque",
        objective="CA par marque",
        family="aggregation",
        tool="run_aggregation",
        proposal={
            "analytical_grain": "brand",
            "group_column": "brand",
            "value_column": "sum_gross_amount",
            "aggregation_function": "sum",
        },
        bindings={
            "group": "brand",
            "value": "sum_gross_amount",
        },
        chart_types={"bar"},
        derived_token=":category:brand:gross_amount",
    ),
    Case(
        name="CA par segment",
        objective="CA par segment",
        family="aggregation",
        tool="run_aggregation",
        proposal={
            "analytical_grain": "segment",
            "group_column": "segment",
            "value_column": "sum_gross_amount",
            "aggregation_function": "sum",
        },
        bindings={
            "group": "segment",
            "value": "sum_gross_amount",
        },
        chart_types={"bar"},
        derived_token=":category:segment:gross_amount",
    ),
    Case(
        name="Top 3 marques par CA",
        objective="Top 3 marques par chiffre d’affaires",
        family="ranking",
        tool="run_ranking",
        proposal={
            "analytical_grain": "brand",
            "dimension_column": "brand",
            "value_column": "sum_gross_amount",
            "aggregation_function": "sum",
            "ranking_order": "descending",
            "ranking_limit": 3,
        },
        bindings={
            "dimension": "brand",
            "value": "sum_gross_amount",
        },
        chart_types={"bar"},
        derived_token=":category:brand:gross_amount",
    ),
    Case(
        name="Évolution mensuelle du CA",
        objective="Évolution mensuelle du chiffre d’affaires",
        family="time_series",
        tool="run_time_series",
        proposal={
            "analytical_grain": "month",
            "time_column": "month",
            "value_column": "sum_gross_amount",
            "aggregation_function": "sum",
        },
        bindings={
            "time": "month",
            "value": "sum_gross_amount",
        },
        chart_types={"line"},
        derived_token=":monthly:order_date:gross_amount",
    ),
    Case(
        name="Panier selon le pays",
        objective="Comment le montant du panier varie-t-il selon le pays ?",
        family="group_comparison",
        tool="run_group_comparison",
        proposal={
            "analytical_grain": "order_id",
            "group_column": "country",
            "value_column": "basket_amount",
            "aggregation_function": "none",
        },
        bindings={
            "group": "country",
            "value": "basket_amount",
        },
        chart_types={"box", "boxplot"},
        result_statuses={"descriptive_only"},
        derived_token=":session:order_id:gross_amount",
    ),
    Case(
        name="Quantité × prix unitaire",
        objective=(
            "Existe-t-il une relation entre la quantité commandée "
            "et le prix unitaire ?"
        ),
        family="quantitative_association",
        tool="run_quantitative_association",
        proposal={
            "x_column": "quantity",
            "y_column": "unit_price",
            "aggregation_function": "none",
        },
        bindings={
            "x": "quantity",
            "y": "unit_price",
        },
        chart_types={"scatter"},
        result_statuses={"needs_specialized_method"},
        source_dataset=True,
    ),
    Case(
        name="Segment × catégorie",
        objective=(
            "Existe-t-il une relation entre le segment client "
            "et la catégorie de produit ?"
        ),
        family="categorical_association",
        tool="run_categorical_association",
        proposal={
            "x_column": "segment",
            "y_column": "category",
            "aggregation_function": "none",
        },
        bindings={
            "x": "segment",
            "y": "category",
        },
        chart_types={"heatmap"},
        result_statuses={"descriptive_only"},
        source_dataset=True,
    ),
)


# ============================================================
# SYNTHETIC SERVER-OWNED PREPARATION OUTPUT
# ============================================================

def build_source_dataframe() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    countries = [
        "France",
        "France",
        "Spain",
        "Germany",
        "France",
        "Spain",
        "Germany",
        "France",
        "Spain",
        "Germany",
        "France",
        "Spain",
        "Germany",
        "France",
        "Spain",
        "Germany",
        "France",
        "Spain",
        "Germany",
        "France",
    ]

    segments = [
        "Consumer",
        "SMB",
        "Enterprise",
        "Consumer",
        "SMB",
        "Enterprise",
        "Consumer",
        "SMB",
        "Enterprise",
        "Consumer",
        "SMB",
        "Enterprise",
        "Consumer",
        "SMB",
        "Enterprise",
        "Consumer",
        "SMB",
        "Enterprise",
        "Consumer",
        "SMB",
    ]

    categories = [
        "Accessories",
        "Electronics",
        "Office",
        "Electronics",
        "Accessories",
        "Office",
        "Electronics",
        "Office",
        "Accessories",
        "Office",
        "Electronics",
        "Accessories",
        "Office",
        "Accessories",
        "Electronics",
        "Office",
        "Accessories",
        "Electronics",
        "Office",
        "Accessories",
    ]

    brands = [
        "Alpha",
        "Beta",
        "Gamma",
        "Alpha",
        "Gamma",
        "Beta",
        "Gamma",
        "Alpha",
        "Beta",
        "Gamma",
        "Alpha",
        "Beta",
        "Gamma",
        "Alpha",
        "Beta",
        "Gamma",
        "Alpha",
        "Beta",
        "Gamma",
        "Alpha",
    ]

    quantities = [
        1, 2, 1, 3, 2,
        1, 4, 1, 2, 3,
        1, 2, 2, 1, 3,
        2, 1, 4, 2, 1,
    ]

    unit_prices = [
        12.0, 35.0, 18.0, 22.0, 15.0,
        42.0, 11.0, 55.0, 16.0, 24.0,
        31.0, 14.0, 28.0, 45.0, 13.0,
        20.0, 38.0, 10.0, 26.0, 17.0,
    ]

    # Repeated customer_id values deliberately model panel/repeated
    # observations so the statistical guards remain exercised.
    customers = [
        "c1", "c1", "c2", "c3", "c4",
        "c2", "c5", "c6", "c3", "c7",
        "c1", "c8", "c4", "c5", "c6",
        "c7", "c2", "c8", "c3", "c4",
    ]

    products = [
        "p1", "p2", "p3", "p4", "p5",
        "p6", "p1", "p2", "p3", "p4",
        "p5", "p6", "p1", "p2", "p3",
        "p4", "p5", "p6", "p1", "p2",
    ]

    for index in range(20):
        month = 1 if index < 10 else 2
        day = (index % 10) + 1

        rows.append(
            {
                "order_id": f"o{index + 1:02d}",
                "customer_id": customers[index],
                "product_id": products[index],
                "order_date": f"2026-{month:02d}-{day:02d}",
                "quantity": quantities[index],
                "unit_price": unit_prices[index],
                "country": countries[index],
                "segment": segments[index],
                "age": 24 + (index % 8) * 4,
                "category": categories[index],
                "brand": brands[index],
            }
        )

    dataframe = pd.DataFrame(
        rows
    )

    # Match the typed Preparation handoff more faithfully.
    #
    # The production Analytical Views layer discovers temporal
    # candidates from the centrally typed DataFrame. Keeping
    # order_date as an object/string in the synthetic fixture
    # prevents the monthly analytical view from being created,
    # even though the column name itself is semantically temporal.
    dataframe[
        "order_date"
    ] = pd.to_datetime(
        dataframe[
            "order_date"
        ],
        errors="raise",
    )

    return dataframe


def build_source_record() -> dict[str, Any]:
    """
    Reproduce the server-owned metadata carried by a validated
    Preparation COMBINE output.

    This metadata is not cosmetic: Analytical Views deliberately
    refuse direct materialization unless preparation_stage is
    present and the record proves that it crossed the trusted
    Preparation -> Analysis handoff boundary.
    """

    return {
        "dataset_id":
            "combine:ci-analysis-prompt-e2e",

        "filename":
            "orders__customers__products.csv",

        "extension":
            ".csv",

        "dataframe":
            build_source_dataframe(),

        "preparation_stage":
            "combine",

        "preparation_workflow_id":
            WORKFLOW_ID,

        "preparation_parent_dataset_ids": [
            "dataset:orders",
            "dataset:customers",
            "dataset:products",
        ],

        "preparation_evidence_refs": [
            "ci:validated-combine",
        ],

        "analysis_input_rule_version":
            "analysis_input_handoff_v0.1",
    }


def fake_handoff(*, workflow_id: str):
    assert workflow_id == WORKFLOW_ID

    return SimpleNamespace(
        ingestion=SimpleNamespace(
            dataset_count=1
        ),
        dataset_records=[
            build_source_record()
        ],
    )


# ============================================================
# CONTROLLED NON-DETERMINISTIC BOUNDARIES
# ============================================================

def _case_for_objective(
    objective: str,
) -> Case:
    normalized = objective.strip()

    for case in CASES:
        if case.objective == normalized:
            return case

    raise AssertionError(
        "No deterministic Gemma wire fixture exists for "
        f"objective={normalized!r}."
    )


def _dataset_id_for_case(
    *,
    case: Case,
    catalog,
) -> str:
    dataset_ids = [
        str(
            dataset.dataset_id
        )
        for dataset
        in catalog.datasets
    ]

    if (
        case.derived_token
        is not None
    ):
        matches = [
            dataset_id
            for dataset_id
            in dataset_ids
            if (
                case.derived_token
                in dataset_id
            )
        ]

        if len(matches) != 1:
            raise AssertionError(
                "Exactly one analytical view must match "
                f"{case.derived_token!r}; found {matches!r}."
            )

        return matches[0]

    source_matches = [
        dataset_id
        for dataset_id
        in dataset_ids
        if not dataset_id.startswith(
            "derived:"
        )
    ]

    if len(source_matches) != 1:
        raise AssertionError(
            "Exactly one source dataset is expected in the "
            f"isolated CI catalog; found {source_matches!r}."
        )

    return source_matches[0]


def fake_generate_raw_ai_plan_with_timing(
    *,
    objective: str,
    catalog,
    model: str = PLANNER_MODEL,
    validation_feedback=None,
):
    """
    Replace only Gemma's inference result.

    Important:
    - intent_routed_planner still decides that these precise
      requests belong to the semantic AI planner;
    - plan_analyses_with_ai() still runs;
    - validate_ai_planner_output() still runs;
    - canonicalization, semantic resolution, abstention guards,
      contract construction and Python validation remain real.

    This is the correct deterministic CI boundary for the nine
    prompts because the current intent router intentionally
    reserves its Python-only generic path for generic outlier
    expansion. These nine analytical questions legitimately use
    the Gemma -> Python-validation path in production.
    """

    del model
    del validation_feedback

    case = _case_for_objective(
        objective
    )

    dataset_id = _dataset_id_for_case(
        case=case,
        catalog=catalog,
    )

    proposal = AIPlannerProposal(
        decision="propose",
        title=case.name,
        family=case.family,
        dataset_id=dataset_id,
        analytical_grain=(
            case.proposal.get(
                "analytical_grain"
            )
        ),
        x_column=(
            case.proposal.get(
                "x_column"
            )
        ),
        y_column=(
            case.proposal.get(
                "y_column"
            )
        ),
        group_column=(
            case.proposal.get(
                "group_column"
            )
        ),
        value_column=(
            case.proposal.get(
                "value_column"
            )
        ),
        time_column=(
            case.proposal.get(
                "time_column"
            )
        ),
        dimension_column=(
            case.proposal.get(
                "dimension_column"
            )
        ),
        entity_column=None,
        aggregation_function=(
            case.proposal.get(
                "aggregation_function",
                "none",
            )
        ),
        ranking_order=(
            case.proposal.get(
                "ranking_order",
                "none",
            )
        ),
        ranking_limit=(
            case.proposal.get(
                "ranking_limit"
            )
        ),
        window_operation="none",
        window_size=None,
        benchmark_reference=None,
        benchmark_operator=None,
        benchmark_selection=None,
        blockers=[],
        reasons=[
            (
                "Deterministic CI fixture for the structured "
                "Gemma wire response. The production Python "
                "planner must still validate this proposal."
            )
        ],
        confidence=0.95,
    )

    return (
        RawAIPlannerOutput(
            proposals=[
                proposal
            ]
        ),
        0.0,
        0.0,
        0.0,
    )


def fake_native_tool_request(
    *,
    contract,
    model: str = TOOL_MODEL,
) -> NativeToolCallRequestResult:
    """
    Replace only the Qwen inference boundary.

    The proposal is generated from the already validated contract
    using the production deterministic helpers. The production
    validate_native_tool_call() still independently validates it,
    and the production deterministic executor still performs the
    actual analysis.
    """

    spec = native_tool_spec_for_contract(
        contract
    )

    expected = expected_tool_arguments(
        contract
    )

    proposal = NativeToolCallProposal(
        tool_name=spec.tool_name,
        arguments=expected.model_dump(),
    )

    attempt = NativeToolCallAttempt(
        attempt_index=1,
        prompt_variant="standard",
        tool_call_count=1,
        assistant_content="",
        selected_tool_name=spec.tool_name,
        errors=[],
        prompt_construction_ms=0.0,
        model_inference_ms=0.0,
        response_parse_ms=0.0,
        total_ms=0.0,
    )

    return NativeToolCallRequestResult(
        proposal=proposal,
        attempts=[attempt],
    )


def fake_register_native_pipeline_result(
    *,
    datasets,
    pipeline_report,
):
    trace_id = str(
        getattr(
            pipeline_report,
            "trace_id",
            "ci",
        )
        or
        "ci"
    )

    return SimpleNamespace(
        analysis_id=f"analysis:{trace_id}",
        source_type="initial_request",
    )


def fake_trace_write(_trace):
    return SimpleNamespace(
        enabled=False,
        written=False,
        error=None,
    )


# ============================================================
# RESPONSE ASSERTIONS
# ============================================================

def require_dict(
    value: Any,
    name: str,
) -> dict[str, Any]:
    if not isinstance(
        value,
        dict,
    ):
        raise AssertionError(
            f"{name} doit être un objet JSON."
        )

    return value


def require_list(
    value: Any,
    name: str,
) -> list[Any]:
    if not isinstance(
        value,
        list,
    ):
        raise AssertionError(
            f"{name} doit être une liste JSON."
        )

    return value


def contract_bindings(
    contract: dict[str, Any],
) -> dict[str, str]:
    result: dict[str, str] = {}

    for raw in require_list(
        contract.get("bindings"),
        "contract.bindings",
    ):
        binding = require_dict(
            raw,
            "binding",
        )

        role = binding.get(
            "role"
        )

        column = binding.get(
            "column"
        )

        if (
            isinstance(role, str)
            and
            isinstance(column, str)
        ):
            result[
                role
            ] = column

    return result


def planner_item(
    body: dict[str, Any],
    family: str,
) -> dict[str, Any]:
    planner = require_dict(
        body.get("planner"),
        "planner",
    )

    matches = []

    for raw in require_list(
        planner.get("items"),
        "planner.items",
    ):
        item = require_dict(
            raw,
            "planner item",
        )

        proposal = item.get(
            "proposal"
        )

        if (
            item.get("validation_status")
            ==
            "validated"
            and
            isinstance(
                proposal,
                dict,
            )
            and
            proposal.get("family")
            ==
            family
        ):
            matches.append(
                item
            )

    if len(
        matches
    ) != 1:
        raise AssertionError(
            "1 contrat planner validé attendu "
            f"pour {family!r}, trouvé: {len(matches)}."
        )

    return matches[
        0
    ]


def pipeline_item(
    body: dict[str, Any],
    family: str,
) -> dict[str, Any]:
    matches = [
        require_dict(
            raw,
            "pipeline item",
        )
        for raw
        in require_list(
            body.get("items"),
            "items",
        )
        if (
            isinstance(
                raw,
                dict,
            )
            and
            raw.get("family")
            ==
            family
        )
    ]

    if len(
        matches
    ) != 1:
        raise AssertionError(
            "1 item pipeline attendu "
            f"pour {family!r}, trouvé: {len(matches)}."
        )

    return matches[
        0
    ]


def run_prompt(
    client: TestClient,
    objective: str,
) -> dict[str, Any]:
    # Multipart/form-data mirrors the browser FormData request.
    response = client.post(
        "/planning/ai-native-run",
        files={
            "workflow_id": (
                None,
                WORKFLOW_ID,
            ),
            "objective": (
                None,
                objective,
            ),
            "planner_model": (
                None,
                PLANNER_MODEL,
            ),
            "tool_model": (
                None,
                TOOL_MODEL,
            ),
        },
    )

    payload = response.json()

    if (
        response.status_code
        !=
        200
    ):
        raise AssertionError(
            f"HTTP {response.status_code}: {payload}"
        )

    return require_dict(
        payload,
        "response",
    )


def validate_case(
    case: Case,
    body: dict[str, Any],
) -> tuple[str, str]:
    assert (
        body.get("status")
        ==
        "ready"
    )

    planner = require_dict(
        body.get("planner"),
        "planner",
    )

    assert (
        planner.get("validated_count")
        ==
        1
    ), (
        "Le planner doit produire exactement 1 contrat validé. "
        f"validated={planner.get('validated_count')}, "
        f"ambiguous={planner.get('ambiguous_count')}, "
        f"blocked={planner.get('blocked_count')}, "
        f"rejected={planner.get('rejected_count')}"
    )

    p_item = planner_item(
        body,
        case.family,
    )

    proposal = require_dict(
        p_item.get("proposal"),
        "proposal",
    )

    for (
        key,
        expected,
    ) in case.proposal.items():
        actual = proposal.get(
            key
        )

        assert (
            actual
            ==
            expected
        ), (
            f"proposal.{key}: "
            f"attendu {expected!r}, "
            f"obtenu {actual!r}"
        )

    dataset_id = proposal.get(
        "dataset_id"
    )

    assert isinstance(
        dataset_id,
        str,
    )

    if (
        case.derived_token
        is not None
    ):
        assert (
            case.derived_token
            in
            dataset_id
        ), (
            f"Mauvaise vue analytique: {dataset_id!r}; "
            f"token attendu: {case.derived_token!r}"
        )

    if (
        case.source_dataset
    ):
        assert not (
            dataset_id.startswith(
                "derived:"
            )
        ), (
            "Une source validée était attendue, "
            f"pas une vue dérivée: {dataset_id!r}"
        )

    contract = require_dict(
        p_item.get("contract"),
        "contract",
    )

    assert (
        contract.get("status")
        ==
        "validated"
    )

    assert (
        contract.get("family")
        ==
        case.family
    )

    actual_bindings = (
        contract_bindings(
            contract
        )
    )

    assert (
        actual_bindings
        ==
        case.bindings
    ), (
        f"Bindings attendus {case.bindings!r}, "
        f"obtenus {actual_bindings!r}"
    )

    pipe = pipeline_item(
        body,
        case.family,
    )

    assert (
        pipe.get("pipeline_status")
        ==
        "executed"
    )

    native = require_dict(
        pipe.get("native_tool"),
        "native_tool",
    )

    assert (
        native.get("validation_status")
        ==
        "validated"
    ), native.get(
        "validation_errors"
    )

    assert (
        native.get("requested_tool")
        ==
        case.tool
    )

    execution = require_dict(
        native.get("execution"),
        "native_tool.execution",
    )

    assert (
        execution.get("execution_status")
        ==
        "executed"
    ), execution.get(
        "errors"
    )

    assert (
        execution.get("tool_name")
        ==
        case.tool
    )

    result = require_dict(
        execution.get("result"),
        "execution.result",
    )

    assert (
        result.get("family")
        ==
        case.family
    )

    result_status = result.get(
        "execution_status"
    )

    assert (
        result_status
        in
        case.result_statuses
    ), (
        f"result.execution_status={result_status!r}, "
        f"attendu={sorted(case.result_statuses)!r}"
    )

    chart_type = result.get(
        "chart_type"
    )

    assert (
        chart_type
        in
        case.chart_types
    ), (
        f"chart_type={chart_type!r}, "
        f"attendu={sorted(case.chart_types)!r}"
    )

    chart_data = result.get(
        "chart_data"
    )

    assert (
        isinstance(
            chart_data,
            list,
        )
        and
        bool(
            chart_data
        )
    ), (
        "chart_data doit être non vide."
    )

    analysis_id = body.get(
        "analysis_id"
    )

    assert (
        isinstance(
            analysis_id,
            str,
        )
        and
        bool(
            analysis_id
        )
    ), (
        "Le pipeline doit exposer un analysis_id serveur."
    )

    return (
        str(
            chart_type
        ),
        str(
            result_status
        ),
    )


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    print(
        "=" * 78
    )
    print(
        "DATALENS ANALYSIS PROMPT DETERMINISTIC E2E REGRESSION v0.4"
    )
    print(
        "=" * 78
    )
    print(
        "HTTP endpoint : /planning/ai-native-run"
    )
    print(
        "Preparation   : isolated validated COMBINE handoff fixture"
    )
    print(
        "Gemma inference: deterministic structured-wire stub"
    )
    print(
        "Qwen inference: deterministic contract-derived stub"
    )
    print(
        "Python engine : real production executor"
    )
    print()

    failures: list[
        tuple[
            str,
            str,
        ]
    ] = []

    with (
        patch(
            "app.api.analysis_run."
            "load_validated_analysis_input_for_http",
            side_effect=
                fake_handoff,
        ),
        patch(
            "app.planning.ai_analytical_planner."
            "_generate_raw_ai_plan_with_timing",
            side_effect=
                fake_generate_raw_ai_plan_with_timing,
        ),
        patch(
            "app.ai.native_tool_calling."
            "request_native_tool_call",
            side_effect=
                fake_native_tool_request,
        ),
        patch(
            "app.ai.ai_native_pipeline."
            "register_native_pipeline_result",
            side_effect=
                fake_register_native_pipeline_result,
        ),
        patch(
            "app.api.analysis_run."
            "write_ai_trace",
            side_effect=
                fake_trace_write,
        ),
    ):
        client = TestClient(
            app
        )

        for (
            index,
            case,
        ) in enumerate(
            CASES,
            start=1,
        ):
            print(
                f"[{index:02d}/{len(CASES):02d}] "
                f"{case.name}"
            )

            try:
                body = run_prompt(
                    client,
                    case.objective,
                )

                (
                    chart_type,
                    result_status,
                ) = validate_case(
                    case,
                    body,
                )

                print(
                    "  [PASS] "
                    f"family={case.family} · "
                    f"tool={case.tool} · "
                    f"chart={chart_type} · "
                    f"result_status={result_status}"
                )

            except Exception as error:
                message = (
                    f"{type(error).__name__}: "
                    f"{error}"
                )

                failures.append(
                    (
                        case.name,
                        message,
                    )
                )

                print(
                    f"  [FAIL] {message}"
                )

            print()

    print(
        "=" * 78
    )

    if (
        failures
    ):
        print(
            "FAIL - "
            f"{len(failures)} / "
            f"{len(CASES)} "
            "cas déterministes en échec"
        )

        for (
            name,
            message,
        ) in failures:
            print(
                f"- {name}: {message}"
            )

        return 1

    print(
        "PASS - 9/9 prompt analyses validated "
        "with deterministic Gemma/Qwen inference boundaries, "
        "real Python validation/execution, and no external backend process"
    )

    return 0


if __name__ == "__main__":
    sys.exit(
        main()
    )

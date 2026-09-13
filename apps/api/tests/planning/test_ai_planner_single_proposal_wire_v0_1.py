from __future__ import annotations

import json

from app.planning.ai_analytical_planner import (
    PlannerCatalog,
    RawAIPlannerOutput,
    normalize_ai_planner_wire_content,
)


RAW_SINGLE_PROPOSAL = """
{
  "decision": "propose",
  "family": "aggregation",
  "dataset_id": "derived:test:category:category:gross_amount",
  "group_column": "category",
  "value_column": "sum_gross_amount",
  "x_column": null,
  "y_column": null,
  "time_column": null,
  "dimension_column": null,
  "entity_column": null,
  "aggregation_function": "sum",
  "ranking_order": "none",
  "ranking_limit": null,
  "window_operation": "none",
  "window_size": null,
  "benchmark_reference": null,
  "benchmark_operator": null,
  "benchmark_selection": null,
  "blockers": []
}
"""


catalog = PlannerCatalog.model_construct(
    datasets=[]
)

normalized = normalize_ai_planner_wire_content(
    content=RAW_SINGLE_PROPOSAL,
    catalog=catalog,
)

decoded = json.loads(
    normalized
)

assert list(decoded.keys()) == [
    "proposals"
]

assert len(
    decoded["proposals"]
) == 1

proposal = decoded[
    "proposals"
][0]

assert proposal["decision"] == "propose"
assert proposal["family"] == "aggregation"

assert (
    proposal["dataset_id"]
    ==
    "derived:test:category:category:gross_amount"
)

assert proposal["group_column"] == "category"

assert (
    proposal["value_column"]
    ==
    "sum_gross_amount"
)

parsed = (
    RawAIPlannerOutput
    .model_validate_json(
        normalized
    )
)

assert len(
    parsed.proposals
) == 1

assert (
    parsed.proposals[0].family
    ==
    "aggregation"
)

print(
    "[PASS] single root planner proposal is wrapped transport-only"
)

print(
    "[PASS] normalized payload satisfies RawAIPlannerOutput"
)

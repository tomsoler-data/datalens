from app.planning.planner_catalog import (
    _analytical_measure_aliases,
)


TRUSTED_MONETARY_EVENT_SEMANTICS = (
    "The unit monetary measure is present "
    "on a server-owned validated Preparation "
    "output produced after controlled "
    "preparation/enrichment. No explicit "
    "quantity measure was detected, so one "
    "prepared event row is conservatively "
    "treated as one monetary event."
)


def build_customer_provenance(
    *,
    source_measure: str = "price",
    metric_semantics: str | None = (
        TRUSTED_MONETARY_EVENT_SEMANTICS
    ),
) -> dict:
    provenance = {
        "operation":
            "customer_behavior_materialization",

        "entity_column":
            "customer_id",

        "source_session_column":
            "session_id",

        "source_measure_column":
            source_measure,

        "target_measure_column":
            "total_spend",

        "grain":
            "customer_id",

        "aggregation_path": [
            "fact rows -> session_id",
            "session_id -> customer_id",
        ],

        "metric_definitions": {
            "total_spend":
                (
                    "Sum of basket_amount "
                    "across observed sessions."
                ),
        },
    }

    if metric_semantics is not None:
        provenance[
            "metric_semantics"
        ] = metric_semantics

    return provenance


# ============================================================
# 1. TRUSTED CUSTOMER REVENUE LINEAGE
# ============================================================

trusted_aliases = set(
    _analytical_measure_aliases(
        provenance=
            build_customer_provenance()
    )
)

assert "price" in trusted_aliases
assert "total_spend" in trusted_aliases

assert "revenue" in trusted_aliases, (
    "Trusted customer total_spend lineage must inherit "
    "the server-owned revenue semantic alias."
)

assert "turnover" in trusted_aliases

assert "chiffre_affaires" in trusted_aliases

assert "ca" in trusted_aliases


# ============================================================
# 2. UNTRUSTED CUSTOMER AGGREGATE MUST FAIL CLOSED
# ============================================================

untrusted_aliases = set(
    _analytical_measure_aliases(
        provenance=
            build_customer_provenance(
                metric_semantics=None
            )
    )
)

assert "revenue" not in untrusted_aliases
assert "turnover" not in untrusted_aliases
assert "chiffre_affaires" not in untrusted_aliases
assert "ca" not in untrusted_aliases


# ============================================================
# 3. NON-REVENUE MONETARY SOURCE MUST NOT BE PROMOTED
# ============================================================

cost_aliases = set(
    _analytical_measure_aliases(
        provenance=
            build_customer_provenance(
                source_measure="cost"
            )
    )
)

assert "revenue" not in cost_aliases
assert "turnover" not in cost_aliases
assert "chiffre_affaires" not in cost_aliases
assert "ca" not in cost_aliases


print(
    "Planner catalog customer revenue alias v0.1: PASS"
)

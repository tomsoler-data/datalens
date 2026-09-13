from __future__ import annotations

from types import SimpleNamespace

from app.api.analysis_run import (
    remove_specialized_entity_outlier_duplicate,
)

from app.planning.ai_analytical_planner import (
    AIPlannerProposal,
    AIPlannerReport,
    AIPlannerValidatedItem,
)


CUSTOMER_DATASET = (
    "derived:test:customer:client_id:price"
)

OTHER_DATASET = (
    "derived:test:other"
)


def make_item(
    index: int,
    family: str,
    dataset_id: str,
):
    proposal = (
        AIPlannerProposal.model_construct(
            title=
                f"proposal {index}",

            family=
                family,

            dataset_id=
                dataset_id,
        )
    )

    return (
        AIPlannerValidatedItem.model_construct(
            proposal_index=
                index,

            validation_status=
                "validated",

            raw_proposal=
                proposal,

            proposal=
                proposal,

            contract=
                None,

            errors=
                [],

            warnings=
                [],

            normalizations=
                [],
        )
    )


def make_report(
    items,
):
    return (
        AIPlannerReport.model_construct(
            objective=
                "test",

            model=
                "test",

            proposal_count=
                len(items),

            validated_count=
                len(items),

            blocked_count=
                0,

            ambiguous_count=
                0,

            rejected_count=
                0,

            items=
                list(items),

            attempt_count=
                1,

            retry_count=
                0,

            retry_triggered=
                False,

            retry_feedback=
                [],

            normalization_count=
                0,

            normalization_applied=
                False,
        )
    )


finding = SimpleNamespace(
    status=
        "ready",

    kind=
        "customer_entity_outlier_detection",

    dataset_id=
        CUSTOMER_DATASET,
)


base_items = [
    make_item(
        1,
        "aggregation",
        "derived:test:scalar",
    ),
    make_item(
        2,
        "ranking",
        CUSTOMER_DATASET,
    ),
    make_item(
        3,
        "distribution",
        CUSTOMER_DATASET,
    ),
]


filtered = (
    remove_specialized_entity_outlier_duplicate(
        planner_report=
            make_report(
                base_items
            ),

        objective=
            (
                "Identifie les 10 meilleurs clients et "
                "signale les ?ventuels clients atypiques."
            ),

        entity_outlier_finding=
            finding,
    )
)


assert (
    filtered.proposal_count
    ==
    2
)

assert [
    item.proposal.family
    for item
    in filtered.items
] == [
    "aggregation",
    "ranking",
]

print(
    "[PASS] duplicate customer distribution is removed"
)


explicit_distribution = (
    remove_specialized_entity_outlier_duplicate(
        planner_report=
            make_report(
                base_items
            ),

        objective=
            (
                "Signale les clients atypiques et montre "
                "la distribution des d?penses clients."
            ),

        entity_outlier_finding=
            finding,
    )
)


assert (
    explicit_distribution.proposal_count
    ==
    3
)

print(
    "[PASS] explicit distribution request is preserved"
)


explicit_value_outlier = (
    remove_specialized_entity_outlier_duplicate(
        planner_report=
            make_report(
                base_items
            ),

        objective=
            (
                "Signale les clients atypiques et les valeurs "
                "atypiques du montant d?pens?."
            ),

        entity_outlier_finding=
            finding,
    )
)


assert (
    explicit_value_outlier.proposal_count
    ==
    3
)

print(
    "[PASS] explicit value-outlier request is preserved"
)


different_dataset_items = [
    base_items[0],
    base_items[1],
    make_item(
        3,
        "distribution",
        OTHER_DATASET,
    ),
]


different_dataset = (
    remove_specialized_entity_outlier_duplicate(
        planner_report=
            make_report(
                different_dataset_items
            ),

        objective=
            "Signale les clients atypiques.",

        entity_outlier_finding=
            finding,
    )
)


assert (
    different_dataset.proposal_count
    ==
    3
)

print(
    "[PASS] unrelated distribution dataset is preserved"
)


without_finding = (
    remove_specialized_entity_outlier_duplicate(
        planner_report=
            make_report(
                base_items
            ),

        objective=
            "Analyse les clients.",

        entity_outlier_finding=
            None,
    )
)


assert (
    without_finding.proposal_count
    ==
    3
)

print(
    "[PASS] planner remains untouched without specialized finding"
)


print()
print(
    "PASS - AI-native entity outlier dedup behavior v0.1"
)

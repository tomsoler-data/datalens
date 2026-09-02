from __future__ import annotations


from app.planning.ai_analytical_planner import (
    AI_ANALYTICAL_PLANNER_RULE_VERSION,
    AIPlannerProposal,
    PlannerCatalog,
    PlannerColumnProfile,
    PlannerDatasetProfile,
    canonicalize_derived_quantitative_association_from_objective,
    objective_schema_column_mentions,
    semantic_metric_match_score,
    validate_ai_proposal,
)


EXPECTED_VERSION = (
    "ai_analytical_planner_v0.35"
)


PROMPT = (
    "Analyse la relation entre l'\u00e2ge au premier achat "
    "des clients et leur panier moyen. "
    "Quantifie la force et le sens de la relation avec une "
    "m\u00e9thode statistique adapt\u00e9e et visualise "
    "le r\u00e9sultat."
)


SOURCE_DATASET_ID = (
    "combine:lapage"
)


CUSTOMER_DATASET_ID = (
    "derived:lapage:customer:client_id:price"
)


def column(
    name: str,
    analysis_kind: str,
    *,
    dtype: str,
    unique_count: int,
) -> PlannerColumnProfile:
    return (
        PlannerColumnProfile(
            name=name,
            dtype=dtype,
            analysis_kind=analysis_kind,
            missing_ratio=0.0,
            unique_count=unique_count,
            unique_candidate=False,
        )
    )


def source_dataset(
) -> PlannerDatasetProfile:
    return (
        PlannerDatasetProfile(
            dataset_id=SOURCE_DATASET_ID,

            filename=(
                "Transactions__customers__products.csv"
            ),

            row_count=687_534,

            column_count=5,

            columns=[
                column(
                    "client_id",
                    "identifier",
                    dtype="object",
                    unique_count=8_621,
                ),

                column(
                    "session_id",
                    "identifier",
                    dtype="object",
                    unique_count=342_315,
                ),

                column(
                    "date",
                    "temporal",
                    dtype="datetime64[ns]",
                    unique_count=687_534,
                ),

                column(
                    "birth",
                    "temporal",
                    dtype="int64",
                    unique_count=76,
                ),

                column(
                    "price",
                    "quantitative",
                    dtype="float64",
                    unique_count=1_455,
                ),
            ],

            is_derived=False,

            analytical_grain="transaction",
        )
    )


def customer_dataset(
    dataset_id: str = CUSTOMER_DATASET_ID,
) -> PlannerDatasetProfile:
    return (
        PlannerDatasetProfile(
            dataset_id=dataset_id,

            filename=(
                dataset_id
                .replace(
                    ":",
                    "_",
                )
                +
                ".derived"
            ),

            row_count=8_621,

            column_count=4,

            columns=[
                column(
                    "client_id",
                    "identifier",
                    dtype="object",
                    unique_count=8_621,
                ),

                # Intentionally place average_basket before age.
                # The canonicalizer must still preserve the order
                # expressed by the user objective.
                column(
                    "average_basket",
                    "quantitative",
                    dtype="float64",
                    unique_count=8_400,
                ),

                column(
                    "total_spend",
                    "quantitative",
                    dtype="float64",
                    unique_count=8_500,
                ),

                column(
                    "age_at_first_purchase",
                    "quantitative",
                    dtype="Float64",
                    unique_count=76,
                ),
            ],

            is_derived=True,

            derivation_type=(
                "entity_additive_measure"
            ),

            analytical_grain=(
                "client_id"
            ),

            operation=(
                "customer_behavior_materialization"
            ),

            entity_column=(
                "client_id"
            ),

            source_measure_column=(
                "price"
            ),

            target_measure_column=(
                "total_spend"
            ),
        )
    )


def catalog(
) -> PlannerCatalog:
    return (
        PlannerCatalog(
            datasets=[
                source_dataset(),
                customer_dataset(),
            ]
        )
    )


def unresolved_proposal(
) -> AIPlannerProposal:
    return (
        AIPlannerProposal(
            decision="propose",

            title=(
                "Association quantitative"
            ),

            family=(
                "quantitative_association"
            ),

            dataset_id=(
                SOURCE_DATASET_ID
            ),

            analytical_grain=(
                "client"
            ),

            x_column=None,
            y_column=None,
            group_column=None,
            value_column=None,
            time_column=None,
            dimension_column=None,
            entity_column=None,

            aggregation_function="none",

            ranking_order="none",
            ranking_limit=None,

            window_operation="none",
            window_size=None,

            benchmark_reference=None,
            benchmark_operator=None,
            benchmark_selection=None,

            blockers=[],
            reasons=[],

            confidence=0.95,
        )
    )


def test_semantic_mapping(
) -> None:
    age_score = (
        semantic_metric_match_score(
            objective=PROMPT,
            column_name=(
                "age_at_first_purchase"
            ),
        )
    )


    basket_score = (
        semantic_metric_match_score(
            objective=PROMPT,
            column_name=(
                "average_basket"
            ),
        )
    )


    assert (
        age_score
        >
        0
    ), age_score


    assert (
        basket_score
        >
        0
    ), basket_score


    mentions = (
        objective_schema_column_mentions(
            objective=PROMPT,
            dataset=(
                customer_dataset()
            ),
        )
    )


    assert (
        "age_at_first_purchase"
        in
        mentions
    ), mentions


    assert (
        "average_basket"
        in
        mentions
    ), mentions


    print(
        "[PASS] French derived-metric semantics"
    )

    print(
        f"       age score={age_score}"
    )

    print(
        f"       basket score={basket_score}"
    )


def test_exact_prompt_is_promoted_and_validated(
) -> None:
    item = (
        validate_ai_proposal(
            objective=PROMPT,

            proposal=(
                unresolved_proposal()
            ),

            proposal_index=1,

            catalog=(
                catalog()
            ),
        )
    )


    assert (
        item.validation_status
        ==
        "validated"
    ), item.model_dump()


    assert (
        item.contract
        is not None
    ), item.model_dump()


    assert (
        item.proposal.family
        ==
        "quantitative_association"
    )


    assert (
        item.proposal.dataset_id
        ==
        CUSTOMER_DATASET_ID
    )


    assert (
        item.proposal.x_column
        ==
        "age_at_first_purchase"
    ), item.model_dump()


    assert (
        item.proposal.y_column
        ==
        "average_basket"
    ), item.model_dump()


    assert (
        item.proposal.analytical_grain
        ==
        "client_id"
    ), item.model_dump()


    bindings = {
        binding.role:
            binding.column

        for binding
        in item.contract.bindings
    }


    assert (
        bindings.get(
            "x"
        )
        ==
        "age_at_first_purchase"
    ), bindings


    assert (
        bindings.get(
            "y"
        )
        ==
        "average_basket"
    ), bindings


    assert any(
        (
            "vue analytique derivee"
            in
            note
        )

        for note
        in item.normalizations
    ), item.normalizations


    print(
        "[PASS] exact Lapage prompt resolves to "
        "existing customer-grain analytical view"
    )

    print(
        "       dataset="
        f"{item.proposal.dataset_id}"
    )

    print(
        "       x="
        f"{item.proposal.x_column}"
    )

    print(
        "       y="
        f"{item.proposal.y_column}"
    )

    print(
        "       grain="
        f"{item.proposal.analytical_grain}"
    )


def test_multiple_matching_derived_views_abstain(
) -> None:
    ambiguous_catalog = (
        PlannerCatalog(
            datasets=[
                source_dataset(),

                customer_dataset(
                    (
                        "derived:lapage:"
                        "customer:client_id:price"
                    )
                ),

                customer_dataset(
                    (
                        "derived:lapage:"
                        "customer:client_id:amount"
                    )
                ),
            ]
        )
    )


    (
        normalized,
        notes,
    ) = (
        canonicalize_derived_quantitative_association_from_objective(
            objective=PROMPT,

            proposal=(
                unresolved_proposal()
            ),

            catalog=(
                ambiguous_catalog
            ),
        )
    )


    assert (
        normalized.dataset_id
        ==
        SOURCE_DATASET_ID
    )


    assert (
        normalized.x_column
        is None
    )


    assert (
        normalized.y_column
        is None
    )


    assert (
        notes
        ==
        []
    )


    item = (
        validate_ai_proposal(
            objective=PROMPT,

            proposal=(
                unresolved_proposal()
            ),

            proposal_index=1,

            catalog=(
                ambiguous_catalog
            ),
        )
    )


    assert (
        item.validation_status
        !=
        "validated"
    ), item.model_dump()


    assert (
        item.contract
        is None
    )


    print(
        "[PASS] multiple derived matches "
        "remain fail-closed"
    )


def test_explicit_dataset_reference_is_not_overridden(
) -> None:
    objective = (
        PROMPT
        +
        " Utilise "
        +
        "Transactions__customers__products.csv."
    )


    (
        normalized,
        notes,
    ) = (
        canonicalize_derived_quantitative_association_from_objective(
            objective=objective,

            proposal=(
                unresolved_proposal()
            ),

            catalog=(
                catalog()
            ),
        )
    )


    assert (
        normalized.dataset_id
        ==
        SOURCE_DATASET_ID
    )


    assert (
        normalized.x_column
        is None
    )


    assert (
        normalized.y_column
        is None
    )


    assert (
        notes
        ==
        []
    )


    print(
        "[PASS] explicit dataset reference "
        "is never overridden"
    )


def test_single_metric_does_not_promote(
) -> None:
    objective = (
        "Analyse le panier moyen des clients."
    )


    (
        normalized,
        notes,
    ) = (
        canonicalize_derived_quantitative_association_from_objective(
            objective=objective,

            proposal=(
                unresolved_proposal()
            ),

            catalog=(
                catalog()
            ),
        )
    )


    assert (
        normalized.dataset_id
        ==
        SOURCE_DATASET_ID
    )


    assert (
        normalized.x_column
        is None
    )


    assert (
        normalized.y_column
        is None
    )


    assert (
        notes
        ==
        []
    )


    print(
        "[PASS] one-metric objective "
        "does not invent an association"
    )


def main() -> None:
    print(
        "=== DATALENS DERIVED QUANTITATIVE "
        "ASSOCIATION v0.1 ==="
    )

    print()


    assert (
        AI_ANALYTICAL_PLANNER_RULE_VERSION
        ==
        EXPECTED_VERSION
    ), (
        AI_ANALYTICAL_PLANNER_RULE_VERSION
    )


    print(
        "[PASS] planner version v0.35"
    )


    test_semantic_mapping()

    test_exact_prompt_is_promoted_and_validated()

    test_multiple_matching_derived_views_abstain()

    test_explicit_dataset_reference_is_not_overridden()

    test_single_metric_does_not_promote()


    print()
    print(
        "=" * 80
    )

    print(
        "FINAL VERDICT"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "French derived metric semantics           PASS"
    )

    print(
        "Unique server-owned derived-view select   PASS"
    )

    print(
        "Quantitative x/y binding                  PASS"
    )

    print(
        "User variable order                       PASS"
    )

    print(
        "Ambiguous derived views                   FAIL-CLOSED"
    )

    print(
        "Explicit dataset authority                PRESERVED"
    )

    print(
        "Single-metric invention guard             PRESERVED"
    )

    print()

    print(
        "V9-P1 DERIVED QUANTITATIVE ASSOCIATION: PASS"
    )


if __name__ == "__main__":
    main()

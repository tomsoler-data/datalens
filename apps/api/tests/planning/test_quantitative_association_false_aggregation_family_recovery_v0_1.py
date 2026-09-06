from __future__ import annotations


from app.planning.ai_analytical_planner import (
    AIPlannerProposal,
    PlannerCatalog,
    PlannerColumnProfile,
    PlannerDatasetProfile,
    canonicalize_derived_quantitative_association_from_objective,
    validate_ai_proposal,
)


EXPECTED_RED_GAP = (
    "quantitative_association_false_aggregation_"
    "family_recovery_gap"
)


RELATION_OBJECTIVE = (
    "What is the relationship between "
    "age_metric and average_basket?"
)


NON_RELATION_OBJECTIVE = (
    "What is the mean average_basket "
    "by age_metric?"
)


def build_dataset(
    dataset_id: str,
) -> PlannerDatasetProfile:

    return PlannerDatasetProfile(
        dataset_id=
            dataset_id,

        filename=
            f"{dataset_id}.derived",

        row_count=
            1000,

        column_count=
            3,

        columns=[
            PlannerColumnProfile(
                name=
                    "entity_id",

                dtype=
                    "object",

                analysis_kind=
                    "identifier",

                missing_ratio=
                    0.0,

                unique_count=
                    1000,

                unique_candidate=
                    True,
            ),

            PlannerColumnProfile(
                name=
                    "age_metric",

                dtype=
                    "float64",

                analysis_kind=
                    "quantitative",

                missing_ratio=
                    0.0,

                unique_count=
                    80,

                unique_candidate=
                    False,
            ),

            PlannerColumnProfile(
                name=
                    "average_basket",

                dtype=
                    "float64",

                analysis_kind=
                    "quantitative",

                missing_ratio=
                    0.0,

                unique_count=
                    900,

                unique_candidate=
                    False,
            ),
        ],

        is_derived=
            True,

        derivation_type=
            "synthetic_entity_behavior",

        analytical_grain=
            "entity",

        operation=
            "entity_behavior_materialization",

        entity_column=
            "entity_id",
    )


def build_catalog() -> PlannerCatalog:

    return PlannerCatalog(
        datasets=[
            build_dataset(
                "derived:synthetic:entity"
            )
        ]
    )


def build_ambiguous_catalog() -> PlannerCatalog:

    return PlannerCatalog(
        datasets=[
            build_dataset(
                "derived:synthetic:entity_a"
            ),
            build_dataset(
                "derived:synthetic:entity_b"
            ),
        ]
    )


def build_proposal(
    *,
    family: str,
) -> AIPlannerProposal:

    return AIPlannerProposal(
        decision=
            "propose",

        title=
            "Synthetic relationship",

        family=
            family,

        dataset_id=
            "derived:synthetic:entity",

        analytical_grain=
            "entity",

        x_column=
            "age_metric",

        y_column=
            "average_basket",

        group_column=
            None,

        value_column=
            None,

        time_column=
            None,

        dimension_column=
            None,

        entity_column=
            None,

        aggregation_function=
            "mean",

        ranking_order=
            "descending",

        ranking_limit=
            None,

        window_operation=
            "none",

        window_size=
            None,

        benchmark_reference=
            "overall_aggregate",

        benchmark_operator=
            "gt",

        benchmark_selection=
            "annotate_all",

        blockers=[],

        reasons=[
            (
                "The objective asks for a relationship "
                "between two quantitative variables."
            )
        ],

        confidence=
            1.0,
    )


def binding_map(
    contract,
) -> dict[str, str]:

    if contract is None:
        return {}


    return {
        str(
            binding.role
        ):
            str(
                binding.column
            )

        for binding
        in contract.bindings
    }


def decision_wires_are_clear(
    proposal: AIPlannerProposal,
) -> bool:

    return bool(
        proposal.aggregation_function
        ==
        "none"

        and

        proposal.ranking_order
        ==
        "none"

        and

        proposal.ranking_limit
        is None

        and

        proposal.benchmark_reference
        is None

        and

        proposal.benchmark_operator
        is None

        and

        proposal.benchmark_selection
        is None

        and

        proposal.group_column
        is None

        and

        proposal.value_column
        is None

        and

        proposal.dimension_column
        is None

        and

        proposal.entity_column
        is None
    )


def direct_recovery_satisfied(
    proposal: AIPlannerProposal,
) -> bool:

    return bool(
        proposal.family
        ==
        "quantitative_association"

        and

        proposal.dataset_id
        ==
        "derived:synthetic:entity"

        and

        proposal.x_column
        ==
        "age_metric"

        and

        proposal.y_column
        ==
        "average_basket"

        and

        decision_wires_are_clear(
            proposal
        )
    )


def main() -> None:

    print(
        "=== DATALENS FALSE AGGREGATION "
        "FAMILY RECOVERY v0.1 ==="
    )

    print()


    catalog = build_catalog()


    # ========================================================
    # 1. ALREADY-CORRECT FAMILY CONTROL
    #
    # Existing behavior must already be GREEN.
    # ========================================================

    correct_family_input = (
        build_proposal(
            family=
                "quantitative_association"
        )
    )


    (
        correct_family_output,
        correct_family_normalizations,
    ) = (
        canonicalize_derived_quantitative_association_from_objective(
            objective=
                RELATION_OBJECTIVE,

            proposal=
                correct_family_input,

            catalog=
                catalog,
        )
    )


    correct_family_control = (
        direct_recovery_satisfied(
            correct_family_output
        )

        and

        len(
            correct_family_normalizations
        )
        >=
        1
    )


    assert correct_family_control is True


    print(
        "[PASS] already-correct quantitative family "
        "uses existing derived-view canonicalizer"
    )


    # ========================================================
    # 2. TARGET:
    # FALSE AGGREGATION FAMILY
    # ========================================================

    false_family_input = (
        build_proposal(
            family=
                "aggregation"
        )
    )


    (
        false_family_direct,
        false_family_normalizations,
    ) = (
        canonicalize_derived_quantitative_association_from_objective(
            objective=
                RELATION_OBJECTIVE,

            proposal=
                false_family_input,

            catalog=
                catalog,
        )
    )


    direct_recovery = (
        direct_recovery_satisfied(
            false_family_direct
        )
    )


    print()

    print(
        "False aggregation direct canonicalization"
    )

    print(
        "  input family                           "
        f"{false_family_input.family}"
    )

    print(
        "  output family                          "
        f"{false_family_direct.family}"
    )

    print(
        "  output x                               "
        f"{false_family_direct.x_column}"
    )

    print(
        "  output y                               "
        f"{false_family_direct.y_column}"
    )

    print(
        "  output aggregation                     "
        f"{false_family_direct.aggregation_function}"
    )

    print(
        "  output ranking                         "
        f"{false_family_direct.ranking_order}"
    )

    print(
        "  output benchmark                       "
        f"{false_family_direct.benchmark_reference}"
    )

    print(
        "  normalizations                         "
        f"{false_family_normalizations}"
    )

    print(
        "  future direct recovery satisfied       "
        f"{direct_recovery}"
    )


    # ========================================================
    # 3. END-TO-END DETERMINISTIC VALIDATOR
    #
    # No LLM execution. We feed the synthetic false wire into
    # the same deterministic validation path used after Gemma.
    # ========================================================

    validated_item = (
        validate_ai_proposal(
            objective=
                RELATION_OBJECTIVE,

            proposal=
                false_family_input,

            proposal_index=
                1,

            catalog=
                catalog,
        )
    )


    contract = (
        validated_item.contract
    )


    bindings = (
        binding_map(
            contract
        )
    )


    end_to_end_recovery = bool(
        validated_item.validation_status
        ==
        "validated"

        and

        contract
        is not None

        and

        contract.family
        ==
        "quantitative_association"

        and

        bindings.get(
            "x"
        )
        ==
        "age_metric"

        and

        bindings.get(
            "y"
        )
        ==
        "average_basket"

        and

        validated_item.proposal.family
        ==
        "quantitative_association"

        and

        decision_wires_are_clear(
            validated_item.proposal
        )
    )


    print()

    print(
        "End-to-end deterministic validation"
    )

    print(
        "  validation status                     "
        f"{validated_item.validation_status}"
    )

    print(
        "  normalized family                     "
        f"{validated_item.proposal.family}"
    )

    print(
        "  normalized aggregation                "
        f"{validated_item.proposal.aggregation_function}"
    )

    print(
        "  normalized ranking                    "
        f"{validated_item.proposal.ranking_order}"
    )

    print(
        "  normalized benchmark                  "
        f"{validated_item.proposal.benchmark_reference}"
    )

    print(
        "  contract family                       "
        f"{contract.family if contract is not None else None}"
    )

    print(
        f"  contract bindings                     {bindings}"
    )

    print(
        "  errors"
    )

    print(
        f"  {list(validated_item.errors)}"
    )

    print(
        "  normalizations"
    )

    print(
        f"  {list(validated_item.normalizations)}"
    )

    print(
        "  future end-to-end recovery satisfied   "
        f"{end_to_end_recovery}"
    )


    # ========================================================
    # 4. NEGATIVE CONTROL:
    # NO RELATIONSHIP CUE
    #
    # A descriptive mean request must not be silently promoted
    # to quantitative association merely because two numeric
    # columns are visible.
    # ========================================================

    (
        non_relation_output,
        non_relation_normalizations,
    ) = (
        canonicalize_derived_quantitative_association_from_objective(
            objective=
                NON_RELATION_OBJECTIVE,

            proposal=
                false_family_input,

            catalog=
                catalog,
        )
    )


    no_relation_preserved = bool(
        non_relation_output.family
        ==
        "aggregation"
    )


    assert no_relation_preserved is True


    print()

    print(
        "[PASS] non-relation objective remains aggregation"
    )


    # ========================================================
    # 5. NEGATIVE CONTROL:
    # AMBIGUOUS DERIVED DATASETS
    #
    # Even with an explicit relationship cue, Python must not
    # invent which derived view to use when several satisfy
    # the same two variables.
    # ========================================================

    ambiguous_input = (
        false_family_input.model_copy(
            update={
                "dataset_id":
                    None,
            }
        )
    )


    (
        ambiguous_output,
        ambiguous_normalizations,
    ) = (
        canonicalize_derived_quantitative_association_from_objective(
            objective=
                RELATION_OBJECTIVE,

            proposal=
                ambiguous_input,

            catalog=
                build_ambiguous_catalog(),
        )
    )


    ambiguous_preserved = bool(
        ambiguous_output.family
        ==
        "aggregation"

        and

        ambiguous_output.dataset_id
        is None
    )


    assert ambiguous_preserved is True


    print(
        "[PASS] ambiguous derived-view match remains fail-closed"
    )


    # ========================================================
    # 6. RED MATRIX
    # ========================================================

    target_recovery = bool(
        direct_recovery
        and
        end_to_end_recovery
    )


    gaps = []


    if not target_recovery:

        gaps.append(
            EXPECTED_RED_GAP
        )


    print()

    print(
        "=" * 80
    )

    print(
        "P7-R1 FALSE AGGREGATION FAMILY RECOVERY MATRIX"
    )

    print(
        "=" * 80
    )

    print()


    print(
        "Correct quantitative family preserved   "
        f"{correct_family_control}"
    )

    print(
        "False aggregation direct recovery        "
        f"{direct_recovery}"
    )

    print(
        "False aggregation end-to-end recovery    "
        f"{end_to_end_recovery}"
    )

    print(
        "Decision wires cleared                  "
        f"{decision_wires_are_clear(validated_item.proposal)}"
    )

    print(
        "Non-relation aggregation preserved       "
        f"{no_relation_preserved}"
    )

    print(
        "Ambiguous derived views fail-closed      "
        f"{ambiguous_preserved}"
    )

    print()

    print(
        f"Observed gaps                            {gaps}"
    )


    if gaps:

        assert (
            gaps
            ==
            [
                EXPECTED_RED_GAP
            ]
        )


        raise AssertionError(
            (
                "RED_EXPECTED: "
                +
                ",".join(
                    gaps
                )
            )
        )


    print()

    print(
        "PASS - false aggregation family "
        "recovery v0.1"
    )


if __name__ == "__main__":

    main()
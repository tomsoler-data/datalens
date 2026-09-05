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
    "quantitative_association_false_ambiguous_"
    "decision_recovery_gap"
)


RELATION_OBJECTIVE = (
    "What is the relationship between "
    "driver_metric and outcome_metric?"
)


NON_RELATION_OBJECTIVE = (
    "What is the mean outcome_metric "
    "by driver_metric?"
)


def column(
    *,
    name: str,
    kind: str,
    unique_count: int,
    unique_candidate: bool = False,
) -> PlannerColumnProfile:

    return PlannerColumnProfile(
        name=
            name,

        dtype=(
            "object"
            if kind == "identifier"
            else "float64"
        ),

        analysis_kind=
            kind,

        missing_ratio=
            0.0,

        unique_count=
            unique_count,

        unique_candidate=
            unique_candidate,
    )


def source_dataset() -> PlannerDatasetProfile:

    return PlannerDatasetProfile(
        dataset_id=
            "source:synthetic",

        filename=
            "source_synthetic.csv",

        row_count=
            5000,

        column_count=
            2,

        columns=[
            column(
                name=
                    "driver_metric",

                kind=
                    "quantitative",

                unique_count=
                    100,
            ),

            column(
                name=
                    "alternate_metric",

                kind=
                    "quantitative",

                unique_count=
                    4500,
            ),
        ],

        is_derived=
            False,

        analytical_grain=
            "event",
    )


def derived_dataset(
    dataset_id: str,
) -> PlannerDatasetProfile:

    return PlannerDatasetProfile(
        dataset_id=
            dataset_id,

        filename=
            (
                dataset_id
                .replace(
                    ":",
                    "_",
                )
                +
                ".derived"
            ),

        row_count=
            1000,

        column_count=
            4,

        columns=[
            column(
                name=
                    "entity_id",

                kind=
                    "identifier",

                unique_count=
                    1000,

                unique_candidate=
                    True,
            ),

            column(
                name=
                    "driver_metric",

                kind=
                    "quantitative",

                unique_count=
                    100,
            ),

            column(
                name=
                    "outcome_metric",

                kind=
                    "quantitative",

                unique_count=
                    900,
            ),

            column(
                name=
                    "alternate_metric",

                kind=
                    "quantitative",

                unique_count=
                    850,
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


def unique_catalog() -> PlannerCatalog:

    return PlannerCatalog(
        datasets=[
            source_dataset(),
            derived_dataset(
                "derived:synthetic:entity"
            ),
        ]
    )


def ambiguous_catalog() -> PlannerCatalog:

    return PlannerCatalog(
        datasets=[
            source_dataset(),
            derived_dataset(
                "derived:synthetic:entity_a"
            ),
            derived_dataset(
                "derived:synthetic:entity_b"
            ),
        ]
    )


def proposal(
    *,
    decision: str,
    blockers: list[str] | None = None,
) -> AIPlannerProposal:

    return AIPlannerProposal(
        decision=
            decision,

        title=
            "Synthetic quantitative relationship",

        family=
            "quantitative_association",

        dataset_id=
            "source:synthetic",

        analytical_grain=
            "entity",

        x_column=
            "driver_metric",

        y_column=
            "alternate_metric",

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

        blockers=(
            list(
                blockers
            )
            if blockers is not None
            else []
        ),

        reasons=[
            (
                "The objective requests a relationship "
                "between two quantitative variables."
            )
        ],

        confidence=
            0.95,
    )


def decision_wires_are_clear(
    value: AIPlannerProposal,
) -> bool:

    return bool(
        value.aggregation_function
        ==
        "none"

        and

        value.ranking_order
        ==
        "none"

        and

        value.ranking_limit
        is None

        and

        value.benchmark_reference
        is None

        and

        value.benchmark_operator
        is None

        and

        value.benchmark_selection
        is None

        and

        value.group_column
        is None

        and

        value.value_column
        is None

        and

        value.time_column
        is None

        and

        value.dimension_column
        is None

        and

        value.entity_column
        is None
    )


def recovered_proposal(
    value: AIPlannerProposal,
) -> bool:

    return bool(
        value.decision
        ==
        "propose"

        and

        value.family
        ==
        "quantitative_association"

        and

        value.dataset_id
        ==
        "derived:synthetic:entity"

        and

        value.analytical_grain
        ==
        "entity"

        and

        value.x_column
        ==
        "driver_metric"

        and

        value.y_column
        ==
        "outcome_metric"

        and

        value.blockers
        ==
        []

        and

        decision_wires_are_clear(
            value
        )
    )


def contract_bindings(
    item,
) -> dict[str, str]:

    if item.contract is None:
        return {}


    return {
        str(
            binding.role
        ):
            str(
                binding.column
            )

        for binding
        in item.contract.bindings
    }


def main() -> None:

    print(
        "=== DATALENS FALSE AMBIGUOUS "
        "DECISION RECOVERY v0.1 ==="
    )

    print()


    catalog = (
        unique_catalog()
    )


    # ========================================================
    # 1. EXISTING PROPOSE PATH
    #
    # Must already be GREEN.
    # ========================================================

    propose_input = (
        proposal(
            decision=
                "propose"
        )
    )


    (
        propose_output,
        propose_normalizations,
    ) = (
        canonicalize_derived_quantitative_association_from_objective(
            objective=
                RELATION_OBJECTIVE,

            proposal=
                propose_input,

            catalog=
                catalog,
        )
    )


    propose_control = bool(
        recovered_proposal(
            propose_output
        )

        and

        len(
            propose_normalizations
        )
        >=
        1
    )


    assert propose_control is True


    print(
        "[PASS] existing propose path remains GREEN"
    )


    # ========================================================
    # 2. TARGET FALSE AMBIGUOUS DECISION
    # ========================================================

    ambiguous_input = (
        proposal(
            decision=
                "ambiguous"
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
                catalog,
        )
    )


    direct_recovery = (
        recovered_proposal(
            ambiguous_output
        )
    )


    print()

    print(
        "False ambiguous direct canonicalization"
    )

    print(
        "  input decision                         "
        f"{ambiguous_input.decision}"
    )

    print(
        "  input blockers                         "
        f"{ambiguous_input.blockers}"
    )

    print(
        "  output decision                        "
        f"{ambiguous_output.decision}"
    )

    print(
        "  output family                          "
        f"{ambiguous_output.family}"
    )

    print(
        "  output dataset                         "
        f"{ambiguous_output.dataset_id}"
    )

    print(
        "  output x                               "
        f"{ambiguous_output.x_column}"
    )

    print(
        "  output y                               "
        f"{ambiguous_output.y_column}"
    )

    print(
        "  output aggregation                     "
        f"{ambiguous_output.aggregation_function}"
    )

    print(
        "  output ranking                         "
        f"{ambiguous_output.ranking_order}"
    )

    print(
        "  output benchmark                       "
        f"{ambiguous_output.benchmark_reference}"
    )

    print(
        "  normalizations                         "
        f"{ambiguous_normalizations}"
    )

    print(
        "  future direct recovery satisfied       "
        f"{direct_recovery}"
    )


    # ========================================================
    # 3. END-TO-END VALIDATION
    # ========================================================

    validated = (
        validate_ai_proposal(
            objective=
                RELATION_OBJECTIVE,

            proposal=
                ambiguous_input,

            proposal_index=
                1,

            catalog=
                catalog,
        )
    )


    bindings = (
        contract_bindings(
            validated
        )
    )


    end_to_end_recovery = bool(
        validated.validation_status
        ==
        "validated"

        and

        recovered_proposal(
            validated.proposal
        )

        and

        validated.contract
        is not None

        and

        validated.contract.status
        ==
        "validated"

        and

        validated.contract.family
        ==
        "quantitative_association"

        and

        bindings.get(
            "x"
        )
        ==
        "driver_metric"

        and

        bindings.get(
            "y"
        )
        ==
        "outcome_metric"

        and

        len(
            validated.errors
        )
        ==
        0
    )


    print()

    print(
        "End-to-end deterministic validation"
    )

    print(
        "  validation status                     "
        f"{validated.validation_status}"
    )

    print(
        "  normalized decision                   "
        f"{validated.proposal.decision}"
    )

    print(
        "  normalized family                     "
        f"{validated.proposal.family}"
    )

    print(
        "  normalized dataset                    "
        f"{validated.proposal.dataset_id}"
    )

    print(
        "  normalized x                          "
        f"{validated.proposal.x_column}"
    )

    print(
        "  normalized y                          "
        f"{validated.proposal.y_column}"
    )

    print(
        "  contract status                       "
        f"{validated.contract.status if validated.contract is not None else None}"
    )

    print(
        "  contract bindings                     "
        f"{bindings}"
    )

    print(
        "  contract blockers                     "
        f"{list(validated.contract.blockers) if validated.contract is not None else []}"
    )

    print(
        "  errors                                "
        f"{list(validated.errors)}"
    )

    print(
        "  normalizations                        "
        f"{list(validated.normalizations)}"
    )

    print(
        "  future end-to-end recovery satisfied   "
        f"{end_to_end_recovery}"
    )


    # ========================================================
    # 4. NEGATIVE CONTROL — BLOCKED
    #
    # Explicit blocked decisions remain authoritative.
    # ========================================================

    blocked_input = (
        proposal(
            decision=
                "blocked",

            blockers=[
                "Required source is unavailable."
            ],
        )
    )


    (
        blocked_output,
        blocked_normalizations,
    ) = (
        canonicalize_derived_quantitative_association_from_objective(
            objective=
                RELATION_OBJECTIVE,

            proposal=
                blocked_input,

            catalog=
                catalog,
        )
    )


    blocked_preserved = bool(
        blocked_output.decision
        ==
        "blocked"

        and

        blocked_output.blockers
        ==
        [
            "Required source is unavailable."
        ]

        and

        len(
            blocked_normalizations
        )
        ==
        0
    )


    assert blocked_preserved is True


    print()

    print(
        "[PASS] blocked decision remains blocked"
    )


    # ========================================================
    # 5. NEGATIVE CONTROL — MEANINGFUL AMBIGUITY
    # ========================================================

    justified_ambiguous_input = (
        proposal(
            decision=
                "ambiguous",

            blockers=[
                (
                    "The requested business concept "
                    "cannot be resolved to one metric."
                )
            ],
        )
    )


    (
        justified_ambiguous_output,
        justified_ambiguous_normalizations,
    ) = (
        canonicalize_derived_quantitative_association_from_objective(
            objective=
                RELATION_OBJECTIVE,

            proposal=
                justified_ambiguous_input,

            catalog=
                catalog,
        )
    )


    justified_ambiguity_preserved = bool(
        justified_ambiguous_output.decision
        ==
        "ambiguous"

        and

        justified_ambiguous_output.blockers
        ==
        justified_ambiguous_input.blockers

        and

        len(
            justified_ambiguous_normalizations
        )
        ==
        0
    )


    assert justified_ambiguity_preserved is True


    print(
        "[PASS] meaningful ambiguous blocker remains fail-closed"
    )


    # ========================================================
    # 6. NEGATIVE CONTROL — NON-RELATION OBJECTIVE
    # ========================================================

    (
        non_relation_output,
        non_relation_normalizations,
    ) = (
        canonicalize_derived_quantitative_association_from_objective(
            objective=
                NON_RELATION_OBJECTIVE,

            proposal=
                ambiguous_input,

            catalog=
                catalog,
        )
    )


    non_relation_preserved = bool(
        non_relation_output.decision
        ==
        "ambiguous"

        and

        len(
            non_relation_normalizations
        )
        ==
        0
    )


    assert non_relation_preserved is True


    print(
        "[PASS] non-relation ambiguous request remains fail-closed"
    )


    # ========================================================
    # 7. NEGATIVE CONTROL — MULTIPLE DERIVED VIEWS
    # ========================================================

    (
        multi_view_output,
        multi_view_normalizations,
    ) = (
        canonicalize_derived_quantitative_association_from_objective(
            objective=
                RELATION_OBJECTIVE,

            proposal=
                ambiguous_input,

            catalog=
                ambiguous_catalog(),
        )
    )


    multiple_views_preserved = bool(
        multi_view_output.decision
        ==
        "ambiguous"

        and

        len(
            multi_view_normalizations
        )
        ==
        0
    )


    assert multiple_views_preserved is True


    print(
        "[PASS] multiple derived views remain fail-closed"
    )


    # ========================================================
    # 8. RED MATRIX
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
        "P7-R3 FALSE AMBIGUOUS DECISION RECOVERY MATRIX"
    )

    print(
        "=" * 80
    )

    print()


    print(
        "Existing propose path                   "
        f"{propose_control}"
    )

    print(
        "False ambiguous direct recovery          "
        f"{direct_recovery}"
    )

    print(
        "False ambiguous end-to-end recovery      "
        f"{end_to_end_recovery}"
    )

    print(
        "Blocked decision preserved               "
        f"{blocked_preserved}"
    )

    print(
        "Meaningful ambiguity preserved           "
        f"{justified_ambiguity_preserved}"
    )

    print(
        "Non-relation ambiguity preserved         "
        f"{non_relation_preserved}"
    )

    print(
        "Multiple views fail-closed               "
        f"{multiple_views_preserved}"
    )

    print()

    print(
        f"Observed gaps                            {gaps}"
    )


    if gaps:

        assert gaps == [
            EXPECTED_RED_GAP
        ]


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
        "PASS - false ambiguous decision "
        "recovery v0.1"
    )


if __name__ == "__main__":

    main()
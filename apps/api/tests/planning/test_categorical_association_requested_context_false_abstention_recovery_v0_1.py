from __future__ import annotations


from app.planning.ai_analytical_planner import (
    AIPlannerProposal,
    PlannerCatalog,
    PlannerColumnProfile,
    PlannerDatasetProfile,
    canonical_semantic_token,
    canonicalize_categorical_association_from_objective,
)


# ============================================================
# OBJECTIVES
# ============================================================


ENGLISH_OBJECTIVE = (
    "Is there an association between gender and category? "
    "Quantify the strength of the association."
)


FRENCH_OBJECTIVE = (
    "Existe-t-il une association entre le genre et la categorie ? "
    "Quantifie l'intensite de cette association."
)


VAGUE_OBJECTIVE = (
    "Analyse the data."
)


MISSING_CATEGORY_OBJECTIVE = (
    "Is there an association between gender and region?"
)


EXPLICIT_DATASET_OBJECTIVE = (
    "In wrong_entity.derived, is there an association "
    "between gender and category?"
)


EXPLICIT_RANKING_OBJECTIVE = (
    "Rank the categories and determine whether there is "
    "an association between gender and category."
)


# ============================================================
# IDS
# ============================================================


SOURCE_DATASET_ID = (
    "dataset:events"
)


REQUESTED_CONTEXT_ID = (
    "derived:events:requested:event_context"
)


WRONG_DATASET_ID = (
    "derived:events:entity:customer_id:amount"
)


SECOND_REQUESTED_CONTEXT_ID = (
    "derived:events:requested:event_context:duplicate"
)


# ============================================================
# EXPECTED RED GAPS
# ============================================================


EXPECTED_GAPS = [
    (
        "categorical_association_"
        "blocked_false_abstention_recovery_gap"
    ),
    (
        "categorical_association_"
        "requested_event_context_recovery_gap"
    ),
    (
        "categorical_association_"
        "gender_bilingual_semantic_gap"
    ),
    (
        "categorical_association_"
        "combined_live_shape_recovery_gap"
    ),
]


# ============================================================
# FIXTURES
# ============================================================


def column(
    *,
    name: str,
    analysis_kind: str,
    dtype: str = "object",
    unique_count: int = 3,
) -> PlannerColumnProfile:

    return PlannerColumnProfile(
        name=name,
        dtype=dtype,
        analysis_kind=analysis_kind,
        missing_ratio=0.0,
        unique_count=unique_count,
        unique_candidate=False,
    )


def source_event_dataset(
    *,
    dataset_id: str = SOURCE_DATASET_ID,
    filename: str = "events.csv",
) -> PlannerDatasetProfile:

    return PlannerDatasetProfile(
        dataset_id=dataset_id,
        filename=filename,
        row_count=1000,
        column_count=3,
        columns=[
            column(
                name="event_id",
                analysis_kind="categorical",
                unique_count=1000,
            ),
            column(
                name="gender",
                analysis_kind="categorical",
                unique_count=2,
            ),
            column(
                name="category",
                analysis_kind="categorical",
                unique_count=3,
            ),
        ],
        is_derived=False,
    )


def requested_event_context(
    *,
    dataset_id: str = REQUESTED_CONTEXT_ID,
    filename: str = "events__requested_event_context.derived",
) -> PlannerDatasetProfile:

    return PlannerDatasetProfile(
        dataset_id=dataset_id,
        filename=filename,
        row_count=1000,
        column_count=4,
        columns=[
            column(
                name="event_id",
                analysis_kind="categorical",
                unique_count=1000,
            ),
            column(
                name="event_time",
                analysis_kind="temporal_datetime",
                dtype="datetime64[ns]",
                unique_count=900,
            ),
            column(
                name="gender",
                analysis_kind="categorical",
                unique_count=2,
            ),
            column(
                name="category",
                analysis_kind="categorical",
                unique_count=3,
            ),
        ],
        is_derived=True,
        derivation_type="requested_event_context",
        analytical_grain="event",
        operation="requested_event_context",
    )


def wrong_derived_context() -> PlannerDatasetProfile:

    return PlannerDatasetProfile(
        dataset_id="derived:events:wrong:event_context",
        filename="events__wrong_context.derived",
        row_count=1000,
        column_count=2,
        columns=[
            column(
                name="gender",
                analysis_kind="categorical",
                unique_count=2,
            ),
            column(
                name="category",
                analysis_kind="categorical",
                unique_count=3,
            ),
        ],
        is_derived=True,
        derivation_type="entity_projection",
        analytical_grain="customer_id",
        operation="projection",
    )


def wrong_entity_dataset() -> PlannerDatasetProfile:

    return PlannerDatasetProfile(
        dataset_id=WRONG_DATASET_ID,
        filename="wrong_entity.derived",
        row_count=100,
        column_count=2,
        columns=[
            column(
                name="customer_id",
                analysis_kind="categorical",
                unique_count=100,
            ),
            column(
                name="total_spend",
                analysis_kind="quantitative",
                dtype="float64",
                unique_count=100,
            ),
        ],
        is_derived=True,
        derivation_type="entity_additive_measure",
        analytical_grain="customer_id",
        operation="groupby_sum",
        aggregation="sum",
        entity_column="customer_id",
        target_measure_column="total_spend",
    )


def planner_proposal(
    *,
    decision: str,
    blocker: str | None = None,
) -> AIPlannerProposal:

    blockers = (
        [
            blocker
        ]
        if blocker is not None
        else
        []
    )


    return AIPlannerProposal(
        decision=decision,
        title="Synthetic categorical association diagnostic",
        family="quantitative_association",
        dataset_id=WRONG_DATASET_ID,
        analytical_grain="customer_id",
        x_column="customer_id",
        y_column="total_spend",
        group_column=None,
        value_column=None,
        time_column=None,
        dimension_column=None,
        entity_column=None,
        aggregation_function="sum",
        ranking_order="descending",
        ranking_limit=None,
        window_operation="none",
        window_size=None,
        benchmark_reference=None,
        benchmark_operator=None,
        benchmark_selection=None,
        blockers=blockers,
        reasons=[
            "Synthetic model output."
        ],
        confidence=0.9,
    )


def false_abstention_proposal() -> AIPlannerProposal:

    return planner_proposal(
        decision="blocked",
        blocker=(
            "The column 'birth' is missing from the dataset. "
            "This column is required for a "
            "quantitative_association."
        ),
    )


def unrelated_blocked_proposal() -> AIPlannerProposal:

    return planner_proposal(
        decision="blocked",
        blocker=(
            "Dataset execution is unavailable because the "
            "underlying source is not accessible."
        ),
    )


# ============================================================
# RECOVERY ASSERTION
# ============================================================


def is_canonical_recovery(
    proposal: AIPlannerProposal,
    *,
    expected_dataset_id: str,
) -> bool:

    return bool(
        proposal.decision
        ==
        "propose"
        and
        proposal.family
        ==
        "categorical_association"
        and
        proposal.dataset_id
        ==
        expected_dataset_id
        and
        {
            proposal.x_column,
            proposal.y_column,
        }
        ==
        {
            "gender",
            "category",
        }
        and
        proposal.group_column
        is None
        and
        proposal.value_column
        is None
        and
        proposal.time_column
        is None
        and
        proposal.dimension_column
        is None
        and
        proposal.entity_column
        is None
        and
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
        proposal.window_operation
        ==
        "none"
        and
        proposal.window_size
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
        proposal.blockers
        ==
        []
    )


def canonicalize(
    *,
    objective: str,
    raw: AIPlannerProposal,
    catalog: PlannerCatalog,
):

    return (
        canonicalize_categorical_association_from_objective(
            objective=objective,
            proposal=raw,
            catalog=catalog,
        )
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    print(
        "=== DATALENS CATEGORICAL ASSOCIATION "
        "THREE-AXIS RECOVERY v0.1 ==="
    )

    print()


    # ========================================================
    # CATALOGS
    # ========================================================

    source_catalog = PlannerCatalog(
        datasets=[
            source_event_dataset(),
            wrong_entity_dataset(),
        ]
    )


    requested_context_catalog = PlannerCatalog(
        datasets=[
            requested_event_context(),
            wrong_entity_dataset(),
        ]
    )


    ambiguous_source_catalog = PlannerCatalog(
        datasets=[
            source_event_dataset(
                dataset_id="dataset:events_a",
                filename="events_a.csv",
            ),
            source_event_dataset(
                dataset_id="dataset:events_b",
                filename="events_b.csv",
            ),
            wrong_entity_dataset(),
        ]
    )


    ambiguous_requested_context_catalog = PlannerCatalog(
        datasets=[
            requested_event_context(),
            requested_event_context(
                dataset_id=SECOND_REQUESTED_CONTEXT_ID,
                filename="events__requested_event_context_2.derived",
            ),
            wrong_entity_dataset(),
        ]
    )


    wrong_derived_catalog = PlannerCatalog(
        datasets=[
            wrong_derived_context(),
            wrong_entity_dataset(),
        ]
    )


    # ========================================================
    # 1. EXISTING BASELINE MUST REMAIN GREEN
    # ========================================================

    (
        baseline,
        baseline_normalizations,
    ) = canonicalize(
        objective=ENGLISH_OBJECTIVE,
        raw=planner_proposal(
            decision="propose",
        ),
        catalog=source_catalog,
    )


    assert (
        is_canonical_recovery(
            baseline,
            expected_dataset_id=SOURCE_DATASET_ID,
        )
    )


    assert (
        len(
            baseline_normalizations
        )
        >=
        1
    )


    print(
        "[PASS] existing source/propose/English recovery"
    )


    # ========================================================
    # 2. FALSE ABSTENTION AXIS
    # ========================================================

    (
        blocked_source,
        _,
    ) = canonicalize(
        objective=ENGLISH_OBJECTIVE,
        raw=false_abstention_proposal(),
        catalog=source_catalog,
    )


    blocked_false_abstention_ok = (
        is_canonical_recovery(
            blocked_source,
            expected_dataset_id=SOURCE_DATASET_ID,
        )
    )


    print(
        "Blocked false-abstention recovery        "
        f"{blocked_false_abstention_ok}"
    )


    # ========================================================
    # 3. REQUESTED EVENT CONTEXT AXIS
    # ========================================================

    (
        derived_propose,
        _,
    ) = canonicalize(
        objective=ENGLISH_OBJECTIVE,
        raw=planner_proposal(
            decision="propose",
        ),
        catalog=requested_context_catalog,
    )


    requested_context_ok = (
        is_canonical_recovery(
            derived_propose,
            expected_dataset_id=REQUESTED_CONTEXT_ID,
        )
    )


    print(
        "Requested-event-context recovery         "
        f"{requested_context_ok}"
    )


    # ========================================================
    # 4. BILINGUAL GENDER AXIS
    # ========================================================

    genre_token = (
        canonical_semantic_token(
            "genre"
        )
    )


    gender_token = (
        canonical_semantic_token(
            "gender"
        )
    )


    categorie_token = (
        canonical_semantic_token(
            "categorie"
        )
    )


    category_token = (
        canonical_semantic_token(
            "category"
        )
    )


    gender_alias_ok = (
        genre_token
        ==
        gender_token
    )


    category_alias_ok = (
        categorie_token
        ==
        category_token
    )


    assert (
        category_alias_ok
    )


    (
        french_source,
        _,
    ) = canonicalize(
        objective=FRENCH_OBJECTIVE,
        raw=planner_proposal(
            decision="propose",
        ),
        catalog=source_catalog,
    )


    french_source_ok = (
        is_canonical_recovery(
            french_source,
            expected_dataset_id=SOURCE_DATASET_ID,
        )
    )


    bilingual_ok = (
        gender_alias_ok
        and
        french_source_ok
    )


    print(
        "genre canonical token                   "
        f"{genre_token}"
    )

    print(
        "gender canonical token                  "
        f"{gender_token}"
    )

    print(
        "genre == gender                         "
        f"{gender_alias_ok}"
    )

    print(
        "categorie == category                   "
        f"{category_alias_ok}"
    )

    print(
        "French categorical recovery             "
        f"{french_source_ok}"
    )


    # ========================================================
    # 5. COMBINED LIVE-SHAPE AXES
    # ========================================================

    (
        combined,
        _,
    ) = canonicalize(
        objective=FRENCH_OBJECTIVE,
        raw=false_abstention_proposal(),
        catalog=requested_context_catalog,
    )


    combined_ok = (
        is_canonical_recovery(
            combined,
            expected_dataset_id=REQUESTED_CONTEXT_ID,
        )
    )


    print(
        "Combined blocked + derived + French      "
        f"{combined_ok}"
    )


    # ========================================================
    # 6. FAIL-CLOSED GUARDS
    # ========================================================

    (
        vague,
        vague_normalizations,
    ) = canonicalize(
        objective=VAGUE_OBJECTIVE,
        raw=planner_proposal(
            decision="propose",
        ),
        catalog=source_catalog,
    )


    assert (
        not is_canonical_recovery(
            vague,
            expected_dataset_id=SOURCE_DATASET_ID,
        )
    )


    assert (
        vague_normalizations
        ==
        []
    )


    print(
        "[PASS] vague objective remains fail-closed"
    )


    (
        absent_category,
        absent_category_normalizations,
    ) = canonicalize(
        objective=MISSING_CATEGORY_OBJECTIVE,
        raw=planner_proposal(
            decision="propose",
        ),
        catalog=source_catalog,
    )


    assert (
        not is_canonical_recovery(
            absent_category,
            expected_dataset_id=SOURCE_DATASET_ID,
        )
    )


    assert (
        absent_category_normalizations
        ==
        []
    )


    print(
        "[PASS] absent requested category remains fail-closed"
    )


    (
        ambiguous_source,
        ambiguous_source_normalizations,
    ) = canonicalize(
        objective=ENGLISH_OBJECTIVE,
        raw=planner_proposal(
            decision="propose",
        ),
        catalog=ambiguous_source_catalog,
    )


    assert (
        ambiguous_source_normalizations
        ==
        []
    )


    assert (
        ambiguous_source.dataset_id
        ==
        WRONG_DATASET_ID
    )


    print(
        "[PASS] ambiguous source candidates remain fail-closed"
    )


    (
        ambiguous_derived,
        ambiguous_derived_normalizations,
    ) = canonicalize(
        objective=ENGLISH_OBJECTIVE,
        raw=planner_proposal(
            decision="propose",
        ),
        catalog=ambiguous_requested_context_catalog,
    )


    assert (
        ambiguous_derived_normalizations
        ==
        []
    )


    assert (
        ambiguous_derived.dataset_id
        ==
        WRONG_DATASET_ID
    )


    print(
        "[PASS] ambiguous requested contexts remain fail-closed"
    )


    (
        wrong_derived,
        wrong_derived_normalizations,
    ) = canonicalize(
        objective=ENGLISH_OBJECTIVE,
        raw=planner_proposal(
            decision="propose",
        ),
        catalog=wrong_derived_catalog,
    )


    assert (
        wrong_derived_normalizations
        ==
        []
    )


    assert (
        wrong_derived.dataset_id
        ==
        WRONG_DATASET_ID
    )


    print(
        "[PASS] unrelated derived view remains fail-closed"
    )


    (
        unrelated_blocked,
        unrelated_blocked_normalizations,
    ) = canonicalize(
        objective=ENGLISH_OBJECTIVE,
        raw=unrelated_blocked_proposal(),
        catalog=source_catalog,
    )


    assert (
        unrelated_blocked.decision
        ==
        "blocked"
    )


    assert (
        unrelated_blocked_normalizations
        ==
        []
    )


    print(
        "[PASS] unrelated blocked reason remains fail-closed"
    )


    (
        explicit_dataset,
        explicit_dataset_normalizations,
    ) = canonicalize(
        objective=EXPLICIT_DATASET_OBJECTIVE,
        raw=false_abstention_proposal(),
        catalog=requested_context_catalog,
    )


    assert (
        explicit_dataset.decision
        ==
        "blocked"
    )


    assert (
        explicit_dataset_normalizations
        ==
        []
    )


    print(
        "[PASS] explicit wrong dataset reference remains fail-closed"
    )


    (
        explicit_ranking,
        explicit_ranking_normalizations,
    ) = canonicalize(
        objective=EXPLICIT_RANKING_OBJECTIVE,
        raw=false_abstention_proposal(),
        catalog=requested_context_catalog,
    )


    assert (
        explicit_ranking.decision
        ==
        "blocked"
    )


    assert (
        explicit_ranking_normalizations
        ==
        []
    )


    print(
        "[PASS] mixed explicit ranking request remains fail-closed"
    )


    # ========================================================
    # 7. EXACT RED
    # ========================================================

    gaps: list[
        str
    ] = []


    if not (
        blocked_false_abstention_ok
    ):

        gaps.append(
            EXPECTED_GAPS[
                0
            ]
        )


    if not (
        requested_context_ok
    ):

        gaps.append(
            EXPECTED_GAPS[
                1
            ]
        )


    if not (
        bilingual_ok
    ):

        gaps.append(
            EXPECTED_GAPS[
                2
            ]
        )


    if not (
        combined_ok
    ):

        gaps.append(
            EXPECTED_GAPS[
                3
            ]
        )


    print()
    print("=" * 80)
    print(
        "P4-R3 THREE-AXIS RECOVERY MATRIX"
    )
    print("=" * 80)
    print()


    print(
        "Existing source/propose/English          True"
    )

    print(
        "Blocked false-abstention recovery        "
        f"{blocked_false_abstention_ok}"
    )

    print(
        "Requested-event-context recovery         "
        f"{requested_context_ok}"
    )

    print(
        "Gender bilingual semantic authority      "
        f"{bilingual_ok}"
    )

    print(
        "Combined live-shape recovery             "
        f"{combined_ok}"
    )

    print()

    print(
        "Observed gaps                            "
        f"{gaps}"
    )


    if gaps:

        raise AssertionError(
            (
                "Categorical association three-axis "
                "recovery gaps: "
                +
                ",".join(
                    gaps
                )
            )
        )


    print()
    print(
        "PASS - categorical association "
        "three-axis recovery v0.1"
    )

    print(
        "P4-R3 REGRESSION: GREEN"
    )


if __name__ == "__main__":

    main()
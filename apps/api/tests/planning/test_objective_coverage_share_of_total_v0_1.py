from __future__ import annotations


import pandas as pd


from app.planning.planner_catalog import (
    planner_catalog_from_dataset_records,
)


from app.planning.objective_coverage import (
    OBJECTIVE_COVERAGE_RULE_VERSION,
    extract_objective_requirements,
)


# ============================================================
# EXPECTED FUTURE OBJECTIVE-COVERAGE VOCABULARY
# ============================================================


SHARE_REQUIREMENT_ID = (
    "derived:share_of_total"
)


SHARE_CONCEPT = (
    "share_of_total"
)


SHARE_REQUIREMENT_TYPE = (
    "derived_metric"
)


# ============================================================
# POSITIVE REQUESTS
# ============================================================


TARGET_OBJECTIVE = (
    "Quelle catégorie contribue le plus au chiffre d'affaires "
    "et quelle est sa part approximative du chiffre d'affaires total ?"
)


FRENCH_PERCENTAGE_OBJECTIVE = (
    "Quelle catégorie génère le plus de chiffre d'affaires "
    "et quel pourcentage du chiffre d'affaires total représente-t-elle ?"
)


ENGLISH_SHARE_OBJECTIVE = (
    "Which category contributes the most revenue and what share "
    "of total revenue does it represent?"
)


# ============================================================
# NEGATIVE GUARDS
# ============================================================


TOTAL_ONLY_OBJECTIVE = (
    "Quel est le chiffre d'affaires total ?"
)


RANKING_ONLY_OBJECTIVE = (
    "Quelle catégorie contribue le plus au chiffre d'affaires ?"
)


BENCHMARK_ONLY_OBJECTIVE = (
    "Quelles catégories ont un chiffre d'affaires "
    "supérieur à la moyenne ?"
)


UNDERSPECIFIED_PART_OBJECTIVE = (
    "Quelle est la part de chaque catégorie ?"
)


# ============================================================
# GENERIC SERVER-OWNED CATALOG
# ============================================================


TRUSTED_MONETARY_EVENT_SEMANTICS = (
    "The unit monetary measure was propagated from a validated "
    "dimension to fact grain. No explicit quantity measure was "
    "detected, so one fact row is conservatively treated as one "
    "monetary event."
)


SOURCE_DATASET_ID = (
    "dataset_source"
)


CATEGORY_DATASET_ID = (
    "derived:dataset_source:category:category:price"
)


def build_catalog():

    source = {
        "dataset_id":
            SOURCE_DATASET_ID,

        "filename":
            "source.csv",

        "dataframe":
            pd.DataFrame(
                {
                    "event_id": [
                        "e1",
                        "e2",
                        "e3",
                    ],

                    "category": [
                        "A",
                        "B",
                        "C",
                    ],

                    "price": [
                        40.0,
                        35.0,
                        25.0,
                    ],
                }
            ),
    }


    category = {
        "dataset_id":
            CATEGORY_DATASET_ID,

        "filename":
            "source__by_category_price.derived",

        "dataframe":
            pd.DataFrame(
                {
                    "category": [
                        "A",
                        "B",
                        "C",
                    ],

                    "sum_price": [
                        40.0,
                        35.0,
                        25.0,
                    ],

                    "event_count": [
                        1,
                        1,
                        1,
                    ],
                }
            ),

        "is_derived":
            True,

        "derivation_type":
            "categorical_additive_measure",

        "source_dataset_ids": [
            SOURCE_DATASET_ID,
        ],

        "provenance": {
            "fact_dataset_id":
                SOURCE_DATASET_ID,

            "operation":
                "groupby_sum",

            "group_column":
                "category",

            "source_measure_column":
                "price",

            "target_measure_column":
                "sum_price",

            "aggregation":
                "sum",

            "grain":
                "category",

            "metric_semantics":
                TRUSTED_MONETARY_EVENT_SEMANTICS,
        },
    }


    return (
        planner_catalog_from_dataset_records(
            [
                source,
                category,
            ]
        )
    )


# ============================================================
# HELPERS
# ============================================================


def requirements_for(
    objective: str,
):

    return (
        extract_objective_requirements(
            objective=
                objective,

            catalog=
                build_catalog(),
        )
    )


def share_requirements_for(
    objective: str,
):

    return [
        requirement

        for requirement
        in requirements_for(
            objective
        )

        if (
            requirement.requirement_id
            ==
            SHARE_REQUIREMENT_ID

            or

            requirement.concept
            ==
            SHARE_CONCEPT
        )
    ]


def revenue_requirement_present(
    objective: str,
) -> bool:

    return any(
        requirement.concept
        ==
        "revenue_total"

        for requirement
        in requirements_for(
            objective
        )
    )


def validate_share_requirement(
    *,
    label: str,
    objective: str,
) -> list[str]:

    gaps: list[str] = []


    matches = (
        share_requirements_for(
            objective
        )
    )


    if (
        len(
            matches
        )
        !=
        1
    ):

        gaps.append(
            f"{label}_share_of_total_requirement_gap"
        )

        return gaps


    requirement = (
        matches[
            0
        ]
    )


    if (
        requirement.requirement_id
        !=
        SHARE_REQUIREMENT_ID
    ):

        gaps.append(
            f"{label}_share_requirement_id_gap"
        )


    if (
        requirement.concept
        !=
        SHARE_CONCEPT
    ):

        gaps.append(
            f"{label}_share_concept_gap"
        )


    if (
        requirement.requirement_type
        !=
        SHARE_REQUIREMENT_TYPE
    ):

        gaps.append(
            f"{label}_share_requirement_type_gap"
        )


    # share_of_total is a derived output requirement.
    # It is deliberately NOT bound to a pre-existing physical
    # catalog column.
    if (
        requirement.candidate_columns
        !=
        []
    ):

        gaps.append(
            f"{label}_share_physical_candidate_gap"
        )


    if (
        requirement.allowed_roles
        !=
        []
    ):

        gaps.append(
            f"{label}_share_role_gap"
        )


    if (
        requirement.required_aggregation
        is not None
    ):

        gaps.append(
            f"{label}_share_aggregation_gap"
        )


    if not (
        requirement.requested_phrases
    ):

        gaps.append(
            f"{label}_share_phrase_evidence_gap"
        )


    return gaps


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    print(
        "=== DATALENS OBJECTIVE COVERAGE "
        "SHARE-OF-TOTAL v0.1 ==="
    )

    print()


    # ========================================================
    # 1. EXISTING REVENUE AUTHORITY MUST REMAIN
    # ========================================================

    assert (
        revenue_requirement_present(
            TARGET_OBJECTIVE
        )
        is True
    )


    assert (
        revenue_requirement_present(
            TOTAL_ONLY_OBJECTIVE
        )
        is True
    )


    print(
        "[PASS] existing revenue_total requirement preserved"
    )


    # ========================================================
    # 2. NEGATIVE GUARDS
    #
    # A share-of-total requirement must be conservative.
    # Neither "total", ranking, benchmark nor an isolated
    # "part" token is sufficient by itself.
    # ========================================================

    negative_cases = {
        "total_only":
            TOTAL_ONLY_OBJECTIVE,

        "ranking_only":
            RANKING_ONLY_OBJECTIVE,

        "benchmark_only":
            BENCHMARK_ONLY_OBJECTIVE,

        "underspecified_part":
            UNDERSPECIFIED_PART_OBJECTIVE,
    }


    for (
        label,
        objective,
    ) in negative_cases.items():

        matches = (
            share_requirements_for(
                objective
            )
        )


        assert (
            matches
            ==
            []
        ), (
            f"{label}: share_of_total must not be inferred "
            "without an explicit numerator/total-share request."
        )


        print(
            f"[PASS] {label} remains non-share"
        )


    # ========================================================
    # 3. POSITIVE SHARE-OF-TOTAL REQUESTS
    # ========================================================

    positive_cases = {
        "target":
            TARGET_OBJECTIVE,

        "french_percentage":
            FRENCH_PERCENTAGE_OBJECTIVE,

        "english_share":
            ENGLISH_SHARE_OBJECTIVE,
    }


    all_gaps: list[str] = []


    print()


    for (
        label,
        objective,
    ) in positive_cases.items():

        requirements = (
            requirements_for(
                objective
            )
        )


        matches = (
            share_requirements_for(
                objective
            )
        )


        print(
            f"{label} requirement IDs"
            f"{' ' * max(1, 28 - len(label))}"
            f"{[
                requirement.requirement_id
                for requirement
                in requirements
            ]}"
        )


        print(
            f"{label} share requirements"
            f"{' ' * max(1, 26 - len(label))}"
            f"{[
                (
                    requirement.requirement_id,
                    requirement.concept,
                    requirement.requirement_type,
                    requirement.candidate_columns,
                )
                for requirement
                in matches
            ]}"
        )


        all_gaps.extend(
            validate_share_requirement(
                label=
                    label,

                objective=
                    objective,
            )
        )


    # ========================================================
    # 4. EXPECTED CURRENT GAP
    # ========================================================

    print()
    print(
        "Objective Coverage rule                 "
        f"{OBJECTIVE_COVERAGE_RULE_VERSION}"
    )

    print(
        "Expected requirement ID                 "
        f"{SHARE_REQUIREMENT_ID}"
    )

    print(
        "Expected concept                        "
        f"{SHARE_CONCEPT}"
    )

    print(
        "Expected requirement type               "
        f"{SHARE_REQUIREMENT_TYPE}"
    )

    print(
        "Expected physical candidates             []"
    )

    print()
    print(
        "Observed gaps                            "
        f"{all_gaps}"
    )


    if all_gaps:

        raise AssertionError(
            (
                "RED_EXPECTED: "
                +
                ",".join(
                    all_gaps
                )
            )
        )


    print()
    print(
        "PASS - Objective Coverage share-of-total v0.1"
    )


if __name__ == "__main__":

    main()
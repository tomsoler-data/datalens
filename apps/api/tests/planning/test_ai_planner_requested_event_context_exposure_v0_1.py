from __future__ import annotations


from unittest.mock import (
    patch,
)


from app.api.analysis_run import (
    prepare_ai_planner_dataset_universe,
)


from app.planning.ai_analytical_planner import (
    PlannerCatalog,
)


# ============================================================
# GENERIC OBJECTIVES
# ============================================================


FRENCH_CATEGORICAL_ASSOCIATION_OBJECTIVE = (
    "Existe-t-il une association entre le genre et la catégorie ? "
    "Quantifie l'intensité de cette association."
)


ENGLISH_CATEGORICAL_ASSOCIATION_OBJECTIVE = (
    "Is there an association between gender and category? "
    "Quantify the strength of the association."
)


QUANTITATIVE_ASSOCIATION_OBJECTIVE = (
    "Quelle est la relation entre l'âge et le montant total ?"
)


AGGREGATION_OBJECTIVE = (
    "Quel est le chiffre d'affaires total ?"
)


VAGUE_OBJECTIVE = (
    "Analyse les données."
)


EXPECTED_GAP = (
    "direct_ai_requested_event_context_exposure_gap"
)


# ============================================================
# POLICY PROBE
# ============================================================


def probe_direct_ai_context_policy(
    objective: str,
) -> bool:

    observed: dict[
        str,
        object,
    ] = {}


    def fake_prepare_analysis_datasets(
        *,
        source_datasets,
        objective,
        include_requested_context=False,
    ):

        observed[
            "source_datasets"
        ] = (
            source_datasets
        )


        observed[
            "objective"
        ] = (
            objective
        )


        observed[
            "include_requested_context"
        ] = (
            include_requested_context
        )


        # prepare_ai_planner_dataset_universe() ignores the
        # Discovery object and uses only analysis_datasets.
        return (
            object(),
            [],
        )


    catalog_sentinel = (
        object()
    )


    with (
        patch(
            "app.api.analysis_run.prepare_analysis_datasets",
            side_effect=
                fake_prepare_analysis_datasets,
        ),
        patch(
            "app.api.analysis_run.planner_catalog_from_dataset_records",
            return_value=
                catalog_sentinel,
        ),
    ):

        (
            analysis_datasets,
            catalog,
        ) = (
            prepare_ai_planner_dataset_universe(
                source_dataset_records=[],
                objective=
                    objective,
            )
        )


    assert (
        analysis_datasets
        ==
        []
    )


    assert (
        catalog
        is
        catalog_sentinel
    )


    assert (
        observed[
            "objective"
        ]
        ==
        objective
    )


    return bool(
        observed[
            "include_requested_context"
        ]
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    print(
        "=== DATALENS DIRECT AI REQUESTED EVENT "
        "CONTEXT EXPOSURE v0.1 ==="
    )

    print()


    # ========================================================
    # 1. POSITIVE — FRENCH CATEGORICAL ASSOCIATION
    # ========================================================

    french_categorical = (
        probe_direct_ai_context_policy(
            FRENCH_CATEGORICAL_ASSOCIATION_OBJECTIVE
        )
    )


    print(
        "French gender/category association       "
        f"{french_categorical}"
    )


    # ========================================================
    # 2. POSITIVE — ENGLISH CATEGORICAL ASSOCIATION
    # ========================================================

    english_categorical = (
        probe_direct_ai_context_policy(
            ENGLISH_CATEGORICAL_ASSOCIATION_OBJECTIVE
        )
    )


    print(
        "English gender/category association      "
        f"{english_categorical}"
    )


    # ========================================================
    # 3. NEGATIVE — QUANTITATIVE ASSOCIATION
    # ========================================================

    quantitative = (
        probe_direct_ai_context_policy(
            QUANTITATIVE_ASSOCIATION_OBJECTIVE
        )
    )


    assert (
        quantitative
        is False
    )


    print(
        "[PASS] unrelated quantitative association "
        "does not request event context"
    )


    # ========================================================
    # 4. NEGATIVE — AGGREGATION
    # ========================================================

    aggregation = (
        probe_direct_ai_context_policy(
            AGGREGATION_OBJECTIVE
        )
    )


    assert (
        aggregation
        is False
    )


    print(
        "[PASS] unrelated aggregation does not "
        "request event context"
    )


    # ========================================================
    # 5. NEGATIVE — VAGUE OBJECTIVE
    # ========================================================

    vague = (
        probe_direct_ai_context_policy(
            VAGUE_OBJECTIVE
        )
    )


    assert (
        vague
        is False
    )


    print(
        "[PASS] vague objective does not request "
        "event context"
    )


    # ========================================================
    # 6. EXACT RED
    # ========================================================

    gaps: list[
        str
    ] = []


    if not (
        french_categorical
        and
        english_categorical
    ):

        gaps.append(
            EXPECTED_GAP
        )


    print()
    print(
        "Expected explicit categorical policy    True"
    )

    print(
        "Observed French policy                  "
        f"{french_categorical}"
    )

    print(
        "Observed English policy                 "
        f"{english_categorical}"
    )

    print(
        "Observed quantitative policy            "
        f"{quantitative}"
    )

    print(
        "Observed aggregation policy             "
        f"{aggregation}"
    )

    print(
        "Observed vague policy                   "
        f"{vague}"
    )

    print()

    print(
        "Observed gaps                           "
        f"{gaps}"
    )


    if gaps:

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
        "PASS - direct AI requested event context "
        "exposure v0.1"
    )


if __name__ == "__main__":

    main()
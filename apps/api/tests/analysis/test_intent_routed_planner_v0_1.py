from __future__ import annotations


import app.planning.intent_routed_planner as routed_module


from app.planning.ai_analytical_planner import (
    AIPlannerReport,
    PlannerCatalog,
    PlannerColumnProfile,
    PlannerDatasetProfile,
)

from app.planning.intent_routed_planner import (
    DETERMINISTIC_GENERIC_PLANNER_MODEL,
    INTENT_ROUTED_PLANNER_RULE_VERSION,
    plan_analyses_with_intent_routing,
)


# ============================================================
# HELPERS
# ============================================================

def column(
    *,
    name: str,
    kind: str,
    dtype: str = "float64",
    unique_count: int = 10,
    unique_candidate: bool = False,
    missing_ratio: float = 0.0,
) -> PlannerColumnProfile:
    return (
        PlannerColumnProfile(
            name=
                name,

            dtype=
                dtype,

            analysis_kind=
                kind,

            missing_ratio=
                missing_ratio,

            unique_count=
                unique_count,

            unique_candidate=
                unique_candidate,
        )
    )


def standard_catalog() -> PlannerCatalog:
    return (
        PlannerCatalog(
            datasets=[
                PlannerDatasetProfile(
                    dataset_id=
                        "dataset:0001",

                    filename=
                        "customers.csv",

                    row_count=
                        100,

                    column_count=
                        4,

                    columns=[
                        column(
                            name=
                                "customer_id",

                            kind=
                                "quantitative",

                            dtype=
                                "int64",

                            unique_count=
                                100,

                            unique_candidate=
                                True,
                        ),

                        column(
                            name=
                                "age",

                            kind=
                                "quantitative",

                            dtype=
                                "int64",

                            unique_count=
                                48,
                        ),

                        column(
                            name=
                                "annual_salary",

                            kind=
                                "quantitative",

                            unique_count=
                                93,
                        ),

                        column(
                            name=
                                "segment",

                            kind=
                                "categorical",

                            dtype=
                                "object",

                            unique_count=
                                4,
                        ),
                    ],
                ),

                PlannerDatasetProfile(
                    dataset_id=
                        "dataset:0002",

                    filename=
                        "sales.csv",

                    row_count=
                        250,

                    column_count=
                        4,

                    columns=[
                        column(
                            name=
                                "order_id",

                            kind=
                                "quantitative",

                            dtype=
                                "int64",

                            unique_count=
                                250,

                            unique_candidate=
                                True,
                        ),

                        column(
                            name=
                                "total_spend",

                            kind=
                                "quantitative",

                            unique_count=
                                221,
                        ),

                        column(
                            name=
                                "discount_rate",

                            kind=
                                "quantitative",

                            unique_count=
                                20,
                        ),

                        column(
                            name=
                                "region",

                            kind=
                                "categorical",

                            dtype=
                                "object",

                            unique_count=
                                5,
                        ),
                    ],
                ),
            ]
        )
    )


def categorical_only_catalog() -> PlannerCatalog:
    return (
        PlannerCatalog(
            datasets=[
                PlannerDatasetProfile(
                    dataset_id=
                        "dataset:0001",

                    filename=
                        "labels.csv",

                    row_count=
                        25,

                    column_count=
                        2,

                    columns=[
                        column(
                            name=
                                "customer_id",

                            kind=
                                "nominal",

                            dtype=
                                "object",

                            unique_count=
                                25,

                            unique_candidate=
                                True,
                        ),

                        column(
                            name=
                                "segment",

                            kind=
                                "categorical",

                            dtype=
                                "object",

                            unique_count=
                                3,
                        ),
                    ],
                ),
            ]
        )
    )


def wide_catalog() -> PlannerCatalog:
    quantitative_columns = [
        column(
            name=
                f"metric_{index:02d}",

            kind=
                "quantitative",

            unique_count=
                50,
        )

        for index
        in range(
            1,
            13,
        )
    ]


    return (
        PlannerCatalog(
            datasets=[
                PlannerDatasetProfile(
                    dataset_id=
                        "dataset:wide",

                    filename=
                        "wide_metrics.csv",

                    row_count=
                        100,

                    column_count=
                        len(
                            quantitative_columns
                        ),

                    columns=
                        quantitative_columns,
                ),
            ]
        )
    )


def empty_fallback_report(
    *,
    objective: str,
) -> AIPlannerReport:
    return (
        AIPlannerReport(
            objective=
                objective,

            model=
                "test:fallback",

            proposal_count=
                0,

            validated_count=
                0,

            blocked_count=
                0,

            ambiguous_count=
                0,

            rejected_count=
                0,

            items=
                [],
        )
    )


# ============================================================
# TEST
# ============================================================

def main() -> None:
    assert (
        INTENT_ROUTED_PLANNER_RULE_VERSION
        ==
        "intent_routed_planner_v0.1"
    )


    # ========================================================
    # 1. GENERIC OUTLIER REQUEST
    #
    # Gemma must NOT be called.
    # ========================================================

    original_fallback = (
        routed_module
        .plan_analyses_with_ai
    )


    fallback_call_count = 0


    def forbidden_fallback(
        *,
        objective: str,
        catalog: PlannerCatalog,
        model: str,
    ) -> AIPlannerReport:
        del objective
        del catalog
        del model


        raise AssertionError(
            "Gemma fallback must not be called "
            "for a deterministic generic outlier request."
        )


    routed_module.plan_analyses_with_ai = (
        forbidden_fallback
    )


    try:
        generic_report = (
            plan_analyses_with_intent_routing(
                objective=
                    "Détecte les outliers.",

                catalog=
                    standard_catalog(),

                model=
                    "gemma3:4b",
            )
        )


    finally:
        routed_module.plan_analyses_with_ai = (
            original_fallback
        )


    assert (
        generic_report.model
        ==
        DETERMINISTIC_GENERIC_PLANNER_MODEL
    )


    assert (
        generic_report.planner_rule_version
        ==
        INTENT_ROUTED_PLANNER_RULE_VERSION
    )


    assert (
        generic_report.proposal_count
        ==
        4
    )


    assert (
        generic_report.validated_count
        ==
        4
    )


    assert (
        generic_report.blocked_count
        ==
        0
    )


    assert (
        generic_report.ambiguous_count
        ==
        0
    )


    assert (
        generic_report.rejected_count
        ==
        0
    )


    assert (
        generic_report.timing.model_inference_ms
        ==
        0.0
    )


    expected_targets = {
        (
            "dataset:0001",
            "age",
        ),
        (
            "dataset:0001",
            "annual_salary",
        ),
        (
            "dataset:0002",
            "total_spend",
        ),
        (
            "dataset:0002",
            "discount_rate",
        ),
    }


    actual_targets: set[
        tuple[
            str,
            str,
        ]
    ] = set()


    for item in (
        generic_report.items
    ):
        assert (
            item.validation_status
            ==
            "validated"
        )


        assert (
            item.contract
            is not None
        )


        contract = (
            item.contract
        )


        assert (
            contract.family
            ==
            "distribution"
        )


        assert (
            contract.status
            ==
            "validated"
        )


        assert (
            contract.request_text
            ==
            "Détecte les outliers."
        )


        assert (
            len(
                contract.required_dataset_ids
            )
            ==
            1
        )


        assert (
            len(
                contract.bindings
            )
            ==
            1
        )


        binding = (
            contract.bindings[
                0
            ]
        )


        assert (
            binding.role
            ==
            "value"
        )


        assert (
            binding.analysis_kind
            ==
            "quantitative"
        )


        actual_targets.add(
            (
                contract.required_dataset_ids[
                    0
                ],
                binding.column,
            )
        )


    assert (
        actual_targets
        ==
        expected_targets
    )


    assert all(
        column_name
        not in {
            "customer_id",
            "order_id",
        }

        for (
            _,
            column_name,
        )
        in actual_targets
    )


    # ========================================================
    # 2. EXPLICIT COLUMN REQUEST
    #
    # The generic resolver must stand aside.
    # Existing AI planner becomes the fallback.
    # ========================================================

    fallback_called = (
        False
    )


    def fake_fallback(
        *,
        objective: str,
        catalog: PlannerCatalog,
        model: str,
    ) -> AIPlannerReport:
        nonlocal fallback_called


        fallback_called = (
            True
        )


        assert (
            objective
            ==
            "Détecte les outliers de annual_salary."
        )


        assert isinstance(
            catalog,
            PlannerCatalog,
        )


        assert (
            model
            ==
            "gemma3:4b"
        )


        return (
            empty_fallback_report(
                objective=
                    objective
            )
        )


    routed_module.plan_analyses_with_ai = (
        fake_fallback
    )


    try:
        explicit_report = (
            plan_analyses_with_intent_routing(
                objective=(
                    "Détecte les outliers "
                    "de annual_salary."
                ),

                catalog=
                    standard_catalog(),

                model=
                    "gemma3:4b",
            )
        )


    finally:
        routed_module.plan_analyses_with_ai = (
            original_fallback
        )


    assert (
        fallback_called
        is True
    )


    assert (
        explicit_report.model
        ==
        "test:fallback"
    )


    # ========================================================
    # 3. UNRELATED REQUEST
    #
    # Also falls back to the existing semantic planner.
    # ========================================================

    unrelated_fallback_called = (
        False
    )


    def fake_unrelated_fallback(
        *,
        objective: str,
        catalog: PlannerCatalog,
        model: str,
    ) -> AIPlannerReport:
        nonlocal unrelated_fallback_called


        unrelated_fallback_called = (
            True
        )


        del catalog
        del model


        return (
            empty_fallback_report(
                objective=
                    objective
            )
        )


    routed_module.plan_analyses_with_ai = (
        fake_unrelated_fallback
    )


    try:
        unrelated_report = (
            plan_analyses_with_intent_routing(
                objective=(
                    "Comparer annual_salary "
                    "selon segment."
                ),

                catalog=
                    standard_catalog(),

                model=
                    "gemma3:4b",
            )
        )


    finally:
        routed_module.plan_analyses_with_ai = (
            original_fallback
        )


    assert (
        unrelated_fallback_called
        is True
    )


    assert (
        unrelated_report.model
        ==
        "test:fallback"
    )


    # ========================================================
    # 4. GENERIC REQUEST WITHOUT QUANTITATIVE TARGET
    #
    # Deterministic abstention. No Gemma call.
    # ========================================================

    routed_module.plan_analyses_with_ai = (
        forbidden_fallback
    )


    try:
        blocked_report = (
            plan_analyses_with_intent_routing(
                objective=
                    "Détecte les valeurs aberrantes.",

                catalog=
                    categorical_only_catalog(),

                model=
                    "gemma3:4b",
            )
        )


    finally:
        routed_module.plan_analyses_with_ai = (
            original_fallback
        )


    assert (
        blocked_report.model
        ==
        DETERMINISTIC_GENERIC_PLANNER_MODEL
    )


    assert (
        blocked_report.proposal_count
        ==
        1
    )


    assert (
        blocked_report.validated_count
        ==
        0
    )


    assert (
        blocked_report.blocked_count
        ==
        1
    )


    assert (
        blocked_report.rejected_count
        ==
        0
    )


    blocked_item = (
        blocked_report.items[
            0
        ]
    )


    assert (
        blocked_item.validation_status
        ==
        "blocked"
    )


    assert (
        blocked_item.contract
        is not None
    )


    assert (
        blocked_item.contract.status
        ==
        "blocked"
    )


    assert (
        blocked_item.contract.family
        ==
        "unresolved"
    )


    assert (
        blocked_item.contract.blockers
    )


    # ========================================================
    # 5. MORE THAN EIGHT DETERMINISTIC TARGETS
    #
    # MAX_AI_PROPOSALS is an LLM protocol limit.
    # It must not truncate catalog-driven deterministic scope.
    # ========================================================

    routed_module.plan_analyses_with_ai = (
        forbidden_fallback
    )


    try:
        wide_report = (
            plan_analyses_with_intent_routing(
                objective=
                    "Find the outliers.",

                catalog=
                    wide_catalog(),

                model=
                    "gemma3:4b",
            )
        )


    finally:
        routed_module.plan_analyses_with_ai = (
            original_fallback
        )


    assert (
        wide_report.proposal_count
        ==
        12
    )


    assert (
        wide_report.validated_count
        ==
        12
    )


    assert (
        wide_report.rejected_count
        ==
        0
    )


    wide_columns = {
        item.contract.bindings[
            0
        ].column

        for item
        in wide_report.items

        if (
            item.contract
            is not None
        )
    }


    assert (
        wide_columns
        ==
        {
            f"metric_{index:02d}"

            for index
            in range(
                1,
                13,
            )
        }
    )


    # ========================================================
    # RESULT
    # ========================================================

    print()
    print(
        "Intent-routed planner v0.1             : OK"
    )

    print(
        "Generic outlier request                : PYTHON"
    )

    print(
        "Gemma bypass for generic request       : OK"
    )

    print(
        "Existing Python validator reused       : OK"
    )

    print(
        "Explicit column request                : AI FALLBACK"
    )

    print(
        "Unrelated analytical request           : AI FALLBACK"
    )

    print(
        "No quantitative target                 : BLOCKED"
    )

    print(
        "> 8 deterministic targets              : SUPPORTED"
    )

    print(
        "Model inference on generic path        : 0 ms"
    )


if __name__ == "__main__":
    main()
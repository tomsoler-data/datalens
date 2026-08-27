from __future__ import annotations


import ast
import inspect


from unittest.mock import (
    patch,
)


from app.api.analysis_run import (
    prepare_ai_planner_dataset_universe,
    preview_ai_analytical_plan,
    run_ai_analytical_tool,
    run_ai_native_pipeline,
)

from app.planning.ai_analytical_planner import (
    PlannerCatalog,
    PlannerColumnProfile,
    PlannerDatasetProfile,
)


# ============================================================
# HELPERS
# ============================================================


def assert_true(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise AssertionError(
            message
        )


def print_pass(
    message: str,
) -> None:
    print(
        f"[PASS] {message}"
    )


def call_keyword_name(
    *,
    function: object,
    called_function_name: str,
    keyword_name: str,
) -> str | None:
    """
    Return the simple variable name passed to one keyword
    argument of a named call inside a function.

    This regression deliberately checks architecture wiring:
    planner catalog construction and deterministic execution
    must share the same `analysis_datasets` universe.
    """

    source = inspect.getsource(
        function
    )


    tree = ast.parse(
        source
    )


    for node in ast.walk(
        tree
    ):
        if not isinstance(
            node,
            ast.Call,
        ):
            continue


        called = (
            node.func.id
            if isinstance(
                node.func,
                ast.Name,
            )
            else None
        )


        if (
            called
            !=
            called_function_name
        ):
            continue


        for keyword in (
            node.keywords
        ):
            if (
                keyword.arg
                !=
                keyword_name
            ):
                continue


            if isinstance(
                keyword.value,
                ast.Name,
            ):
                return (
                    keyword.value.id
                )


            return None


    return None


def function_calls_name(
    *,
    function: object,
    called_function_name: str,
) -> bool:
    source = inspect.getsource(
        function
    )


    tree = ast.parse(
        source
    )


    for node in ast.walk(
        tree
    ):
        if not isinstance(
            node,
            ast.Call,
        ):
            continue


        if (
            isinstance(
                node.func,
                ast.Name,
            )
            and
            node.func.id
            ==
            called_function_name
        ):
            return True


    return False


def build_stub_catalog() -> PlannerCatalog:
    return PlannerCatalog(
        datasets=[
            PlannerDatasetProfile(
                dataset_id=(
                    "derived:dataset_demo:"
                    "category:category:gross_amount"
                ),
                filename=(
                    "demo__by_category_gross_amount.derived"
                ),
                row_count=3,
                column_count=3,
                columns=[
                    PlannerColumnProfile(
                        name="category",
                        dtype="object",
                        analysis_kind="categorical",
                        missing_ratio=0.0,
                        unique_count=3,
                        unique_candidate=False,
                    ),
                    PlannerColumnProfile(
                        name="sum_gross_amount",
                        dtype="float64",
                        analysis_kind="quantitative",
                        missing_ratio=0.0,
                        unique_count=3,
                        unique_candidate=False,
                    ),
                    PlannerColumnProfile(
                        name="event_count",
                        dtype="int64",
                        analysis_kind="quantitative",
                        missing_ratio=0.0,
                        unique_count=2,
                        unique_candidate=False,
                    ),
                ],
                is_derived=True,
                derivation_type=(
                    "categorical_additive_measure"
                ),
                analytical_grain="category",
                operation="groupby_sum",
                aggregation="sum",
                group_column="category",
                entity_column=None,
                source_measure_column="gross_amount",
                target_measure_column="sum_gross_amount",
                source_measure_formula=(
                    "quantity * unit_price"
                ),
                metric_semantics=(
                    "Controlled monetary analytical measure."
                ),
                measure_semantic_aliases=[
                    "gross_amount",
                    "sum_gross_amount",
                    "revenue",
                    "chiffre_affaires",
                    "ca",
                ],
            )
        ]
    )


# ============================================================
# TESTS
# ============================================================


def test_helper_builds_catalog_from_analysis_datasets() -> None:
    source_record = {
        "dataset_id":
            "dataset:demo",

        "filename":
            "demo.csv",

        "dataframe":
            object(),
    }


    derived_record = {
        "dataset_id":
            (
                "derived:dataset_demo:"
                "category:category:gross_amount"
            ),

        "filename":
            "demo__by_category_gross_amount.derived",

        "dataframe":
            object(),

        "is_derived":
            True,

        "derivation_type":
            "categorical_additive_measure",

        "provenance":
            {
                "grain":
                    "category",
            },
    }


    analysis_datasets = [
        source_record,
        derived_record,
    ]


    expected_catalog = (
        build_stub_catalog()
    )


    catalog_inputs: list[
        list[
            dict[
                str,
                object,
            ]
        ]
    ] = []


    def fake_catalog_builder(
        records,
    ):
        catalog_inputs.append(
            records
        )

        return (
            expected_catalog
        )


    with patch(
        "app.api.analysis_run.prepare_analysis_datasets",
        return_value=(
            object(),
            analysis_datasets,
        ),
    ) as prepare_mock:
        with patch(
            "app.api.analysis_run.planner_catalog_from_dataset_records",
            side_effect=
                fake_catalog_builder,
        ):
            (
                returned_datasets,
                returned_catalog,
            ) = prepare_ai_planner_dataset_universe(
                source_dataset_records=[
                    source_record
                ],
                objective="CA par catégorie",
            )


    assert_true(
        returned_datasets
        is
        analysis_datasets,
        (
            "The helper must return the exact analytical "
            "dataset universe produced by "
            "prepare_analysis_datasets()."
        ),
    )


    assert_true(
        returned_catalog
        is
        expected_catalog,
        (
            "The helper must return the planner catalog built "
            "from that same analytical universe."
        ),
    )


    assert_true(
        catalog_inputs
        ==
        [
            analysis_datasets
        ],
        (
            "planner_catalog_from_dataset_records() must receive "
            "analysis_datasets, not source_dataset_records."
        ),
    )


    prepare_mock.assert_called_once_with(
        source_datasets=[
            source_record
        ],
        objective="CA par catégorie",
        include_requested_context=False,
    )


    print_pass(
        "AI planner helper builds catalog from the exact analytical dataset universe"
    )


def test_preview_uses_shared_ai_dataset_universe() -> None:
    assert_true(
        function_calls_name(
            function=
                preview_ai_analytical_plan,
            called_function_name=
                "prepare_ai_planner_dataset_universe",
        ),
        (
            "AI planner preview must prepare the shared "
            "analytical dataset universe."
        ),
    )


    assert_true(
        call_keyword_name(
            function=
                preview_ai_analytical_plan,
            called_function_name=
                "plan_analyses_with_intent_routing",
            keyword_name=
                "catalog",
        )
        ==
        "catalog",
        (
            "AI planner preview must plan from the catalog "
            "returned by the shared analytical-universe helper."
        ),
    )


    print_pass(
        "AI planner preview uses the shared analytical dataset universe"
    )


def test_tool_orchestration_executes_same_universe_used_by_planner() -> None:
    assert_true(
        function_calls_name(
            function=
                run_ai_analytical_tool,
            called_function_name=
                "prepare_ai_planner_dataset_universe",
        ),
        (
            "AI tool orchestration must prepare the shared "
            "analytical dataset universe."
        ),
    )


    assert_true(
        call_keyword_name(
            function=
                run_ai_analytical_tool,
            called_function_name=
                "execute_ai_planner_report",
            keyword_name=
                "datasets",
        )
        ==
        "analysis_datasets",
        (
            "AI tool orchestration must execute against "
            "analysis_datasets so a validated derived dataset "
            "contract remains executable."
        ),
    )


    print_pass(
        "AI tool orchestration executes the same analytical universe used by the planner"
    )


def test_native_pipeline_executes_same_universe_used_by_planner() -> None:
    assert_true(
        function_calls_name(
            function=
                run_ai_native_pipeline,
            called_function_name=
                "prepare_ai_planner_dataset_universe",
        ),
        (
            "AI native pipeline must prepare the shared "
            "analytical dataset universe."
        ),
    )


    assert_true(
        call_keyword_name(
            function=
                run_ai_native_pipeline,
            called_function_name=
                "execute_native_ai_pipeline",
            keyword_name=
                "datasets",
        )
        ==
        "analysis_datasets",
        (
            "AI native executor must receive analysis_datasets, "
            "not source_dataset_records."
        ),
    )


    print_pass(
        "AI native pipeline executes the same analytical universe used by the planner"
    )


def main() -> None:
    print(
        "=== DATALENS AI PLANNER ANALYTICAL UNIVERSE v0.1 ==="
    )


    print()


    test_helper_builds_catalog_from_analysis_datasets()

    test_preview_uses_shared_ai_dataset_universe()

    test_tool_orchestration_executes_same_universe_used_by_planner()

    test_native_pipeline_executes_same_universe_used_by_planner()


    print()
    print(
        "PASS - AI planner analytical universe v0.1"
    )


if __name__ == "__main__":
    main()

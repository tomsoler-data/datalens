from __future__ import annotations

from app.ai.native_tool_calling import (
    EntityOutlierToolArgs,
    NATIVE_TOOL_CALLING_RULE_VERSION,
    NATIVE_TOOL_SPECS,
    NativeToolCallProposal,
    build_native_tool_schema,
    expected_tool_arguments,
    validate_native_tool_call,
)

from app.planning.analytical_contract import (
    AnalyticalContract,
)


def build_contract() -> AnalyticalContract:
    return AnalyticalContract.model_validate(
        {
            "contract_id":
                "ai:test-entity-outlier:01",

            "contract_version":
                "analytical_contract_v0.4",

            "origin":
                "ai_planner",

            "status":
                "validated",

            "title":
                "Supplier delivery-delay outliers",

            "request_text":
                (
                    "Identifie les fournisseurs atypiques "
                    "selon delivery_delay_days."
                ),

            "family":
                "entity_outlier",

            "required_dataset_ids":
                [
                    "dataset:suppliers"
                ],

            "required_dataset_filenames":
                [
                    "suppliers.csv"
                ],

            "analytical_grain":
                "supplier_id",

            "bindings":
                [
                    {
                        "role":
                            "entity",

                        "column":
                            "supplier_id",

                        "dataset_id":
                            "dataset:suppliers",

                        "dataset_filename":
                            "suppliers.csv",

                        "semantic_concept":
                            None,

                        "analysis_kind":
                            "categorical",
                    },

                    {
                        "role":
                            "value",

                        "column":
                            "delivery_delay_days",

                        "dataset_id":
                            "dataset:suppliers",

                        "dataset_filename":
                            "suppliers.csv",

                        "semantic_concept":
                            None,

                        "analysis_kind":
                            "quantitative",
                    },
                ],

            "aggregation":
                None,

            "ranking":
                None,

            "benchmark":
                None,

            "share_of_total":
                None,

            "window":
                None,

            "filters":
                [],

            "joins":
                [],

            "derived_variables":
                [],

            "required_operations":
                [
                    (
                        "Execute only through a deterministic "
                        "entity_outlier tool."
                    )
                ],

            "provenance":
                None,

            "reasons":
                [
                    (
                        "The LLM selected the analytical family "
                        "and exact column roles."
                    )
                ],

            "blockers":
                [],

            "planner_confidence":
                None,
        }
    )


def test_native_tool_version_v0_10() -> None:
    assert (
        NATIVE_TOOL_CALLING_RULE_VERSION
        ==
        "native_tool_calling_v0.10"
    )


def test_entity_outlier_tool_is_registered() -> None:
    spec = (
        NATIVE_TOOL_SPECS[
            "entity_outlier"
        ]
    )

    assert (
        spec.tool_name
        ==
        "run_entity_outlier"
    )

    assert (
        spec.argument_shape
        ==
        "entity_value"
    )


def test_entity_outlier_schema_has_exact_arguments() -> None:
    spec = (
        NATIVE_TOOL_SPECS[
            "entity_outlier"
        ]
    )

    schema = (
        build_native_tool_schema(
            spec
        )
    )

    parameters = (
        schema[
            "function"
        ][
            "parameters"
        ]
    )

    assert set(
        parameters[
            "required"
        ]
    ) == {
        "dataset_id",
        "entity_column",
        "value_column",
    }

    assert set(
        parameters[
            "properties"
        ]
    ) == {
        "dataset_id",
        "entity_column",
        "value_column",
    }

    assert (
        parameters[
            "additionalProperties"
        ]
        is False
    )


def test_expected_entity_outlier_args_copy_contract_exactly() -> None:
    expected = (
        expected_tool_arguments(
            build_contract()
        )
    )

    assert isinstance(
        expected,
        EntityOutlierToolArgs,
    )

    assert (
        expected.model_dump()
        ==
        {
            "dataset_id":
                "dataset:suppliers",

            "entity_column":
                "supplier_id",

            "value_column":
                "delivery_delay_days",
        }
    )


def test_exact_entity_outlier_tool_call_is_valid() -> None:
    errors = (
        validate_native_tool_call(
            contract=
                build_contract(),

            proposal=
                NativeToolCallProposal(
                    tool_name=
                        "run_entity_outlier",

                    arguments=
                        {
                            "dataset_id":
                                "dataset:suppliers",

                            "entity_column":
                                "supplier_id",

                            "value_column":
                                "delivery_delay_days",
                        },
                ),
        )
    )

    assert errors == []


def test_entity_outlier_wrong_entity_is_rejected() -> None:
    errors = (
        validate_native_tool_call(
            contract=
                build_contract(),

            proposal=
                NativeToolCallProposal(
                    tool_name=
                        "run_entity_outlier",

                    arguments=
                        {
                            "dataset_id":
                                "dataset:suppliers",

                            "entity_column":
                                "region",

                            "value_column":
                                "delivery_delay_days",
                        },
                ),
        )
    )

    assert errors
    assert (
        "entity_column"
        in
        " ".join(
            errors
        )
    )


def test_entity_outlier_wrong_value_is_rejected() -> None:
    errors = (
        validate_native_tool_call(
            contract=
                build_contract(),

            proposal=
                NativeToolCallProposal(
                    tool_name=
                        "run_entity_outlier",

                    arguments=
                        {
                            "dataset_id":
                                "dataset:suppliers",

                            "entity_column":
                                "supplier_id",

                            "value_column":
                                "defect_rate",
                        },
                ),
        )
    )

    assert errors
    assert (
        "value_column"
        in
        " ".join(
            errors
        )
    )


def test_entity_outlier_extra_argument_is_rejected() -> None:
    errors = (
        validate_native_tool_call(
            contract=
                build_contract(),

            proposal=
                NativeToolCallProposal(
                    tool_name=
                        "run_entity_outlier",

                    arguments=
                        {
                            "dataset_id":
                                "dataset:suppliers",

                            "entity_column":
                                "supplier_id",

                            "value_column":
                                "delivery_delay_days",

                            "metric_selected_by_python":
                                "forbidden",
                        },
                ),
        )
    )

    assert errors

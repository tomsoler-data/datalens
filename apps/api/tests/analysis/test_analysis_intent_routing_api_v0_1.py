from __future__ import annotations


from io import (
    StringIO,
)


from unittest.mock import (
    patch,
)


import pandas as pd


from fastapi.testclient import (
    TestClient,
)


from main import (
    app,
)


from app.preparation.preparation_artifact_store import (
    put_preparation_artifact,
    reset_preparation_artifact_store_for_tests,
)


from app.preparation.preparation_session import (
    create_preparation_session,
    get_preparation_session,
    record_analysis_output_selection,
    record_required_stage_signal,
    record_validation_stage_signal,
    reset_preparation_session_store_for_tests,
)


from app.preparation.preparation_workflow import (
    PreparationStage,
)


# ============================================================
# CLIENT
# ============================================================


client = TestClient(
    app
)


# ============================================================
# CSV FIXTURE
# ============================================================


def build_employees_csv(
) -> str:
    rows = [
        (
            "customer_id,"
            "age,"
            "annual_salary,"
            "satisfaction_score,"
            "department"
        )
    ]


    departments = [
        "Sales",
        "IT",
        "Finance",
        "Operations",
    ]


    for index in range(
        1,
        41,
    ):
        customer_id = (
            f"C{index:03d}"
        )


        age = (
            22
            +
            (
                index
                %
                12
            )
        )


        annual_salary = (
            35000
            +
            (
                index
                %
                8
            )
            *
            5000
        )


        satisfaction_score = (
            2.5
            +
            (
                index
                %
                6
            )
            *
            0.5
        )


        department = (
            departments[
                index
                %
                len(
                    departments
                )
            ]
        )


        rows.append(
            (
                f"{customer_id},"
                f"{age},"
                f"{annual_salary},"
                f"{satisfaction_score},"
                f"{department}"
            )
        )


    return "\n".join(
        rows
    )



# ============================================================
# SERVER-OWNED PREPARATION FIXTURE
# ============================================================


def prepare_validated_workflow(
    *,
    csv_content: str,
    dataset_filename: str,
) -> str:

    reset_preparation_session_store_for_tests()
    reset_preparation_artifact_store_for_tests()


    dataset_id = (
        "dataset:0001"
    )


    session = (
        create_preparation_session(
            selected_analysis_dataset_ids=[
                dataset_id
            ]
        )
    )


    workflow_id = (
        session.workflow_id
    )


    for (
        stage,
        evidence_ref,
    ) in (
        (
            PreparationStage.IMPORT,
            "csv_ingestion",
        ),
        (
            PreparationStage.UNDERSTAND,
            "dataset_profile",
        ),
        (
            PreparationStage.QUALITY,
            "data_quality_engine_v0.2",
        ),
    ):
        record_required_stage_signal(
            workflow_id=
                workflow_id,

            stage=
                stage,

            completed=
                True,

            dataset_ids=[
                dataset_id
            ],

            evidence_refs=[
                evidence_ref
            ],

            blocking_reasons=[],
        )


    dataframe = (
        pd.read_csv(
            StringIO(
                csv_content
            )
        )
    )


    put_preparation_artifact(
        workflow_id=
            workflow_id,

        dataset_id=
            dataset_id,

        dataset_filename=
            dataset_filename,

        stage=
            "source",

        dataframe=
            dataframe,

        parent_dataset_ids=[],

        evidence_refs=[
            "test:server-owned-planning-fixture"
        ],
    )


    before_selection = (
        get_preparation_session(
            workflow_id
        )
    )


    selected = (
        record_analysis_output_selection(
            workflow_id=
                workflow_id,

            analysis_output_dataset_ids=[
                dataset_id
            ],

            expected_revision=
                before_selection.revision,
        )
    )


    ready = (
        record_validation_stage_signal(
            workflow_id=
                workflow_id,

            completed=
                True,

            passed=
                True,

            dataset_ids=[
                dataset_id
            ],

            evidence_refs=[
                "final_validation"
            ],

            blocking_reasons=[],

            expected_revision=
                selected.revision,
        )
    )


    assert (
        ready
        .snapshot
        .ready_for_analysis
        is True
    )


    return (
        workflow_id
    )


# ============================================================
# GEMMA MUST NEVER RUN
# ============================================================


def forbidden_ai_fallback(
    *args,
    **kwargs,
):
    del args
    del kwargs


    raise AssertionError(
        (
            "Gemma must not be called for the "
            "generic outlier request."
        )
    )


# ============================================================
# TEST
# ============================================================


def test_generic_outlier_preview_uses_python_router(
) -> None:
    csv_content = (
        build_employees_csv()
    )


    workflow_id = (
        prepare_validated_workflow(
            csv_content=
                csv_content,

            dataset_filename=
                "employees.csv",
        )
    )


    with patch(
        (
            "app.planning."
            "intent_routed_planner."
            "plan_analyses_with_ai"
        ),

        side_effect=
            forbidden_ai_fallback,
    ):
        response = (
            client.post(
                "/planning/ai-preview",

                data={
                    "workflow_id":
                        workflow_id,

                    "objective":
                        "Détecte les outliers.",

                    "planner_model":
                        "gemma3:4b",
                },

            )
        )


    print(
        "\n=== GENERIC OUTLIER API ==="
    )

    print(
        f"Status: "
        f"{response.status_code}"
    )


    assert (
        response.status_code
        ==
        200
    )


    body = (
        response.json()
    )


    print(
        f"Planner model: "
        f"{body['model']}"
    )

    print(
        (
            "Proposals: "
            f"{body['proposal_count']}"
        )
    )

    print(
        (
            "Validated: "
            f"{body['validated_count']}"
        )
    )

    print(
        (
            "Rejected: "
            f"{body['rejected_count']}"
        )
    )

    print(
        (
            "Model inference: "
            f"{body['timing']['model_inference_ms']} ms"
        )
    )


    # ========================================================
    # ROUTING
    # ========================================================

    assert (
        body[
            "model"
        ]
        ==
        (
            "python:"
            "generic_analytical_intent_v0.1"
        )
    )


    assert (
        body[
            "planner_rule_version"
        ]
        ==
        "intent_routed_planner_v0.1"
    )


    assert (
        body[
            "timing"
        ][
            "model_inference_ms"
        ]
        ==
        0.0
    )


    # ========================================================
    # RESULT COUNTS
    # ========================================================

    assert (
        body[
            "proposal_count"
        ]
        >
        0
    )


    assert (
        body[
            "validated_count"
        ]
        >
        0
    )


    assert (
        body[
            "blocked_count"
        ]
        ==
        0
    )


    assert (
        body[
            "ambiguous_count"
        ]
        ==
        0
    )


    assert (
        body[
            "rejected_count"
        ]
        ==
        0
    )


    # ========================================================
    # CONTRACTS
    # ========================================================

    validated_items = [
        item

        for item
        in body[
            "items"
        ]

        if (
            item[
                "validation_status"
            ]
            ==
            "validated"
        )
    ]


    assert (
        len(
            validated_items
        )
        ==
        body[
            "validated_count"
        ]
    )


    target_columns: set[
        str
    ] = set()


    for item in (
        validated_items
    ):
        contract = (
            item[
                "contract"
            ]
        )


        assert (
            contract
            is not None
        )


        assert (
            contract[
                "family"
            ]
            ==
            "distribution"
        )


        assert (
            contract[
                "status"
            ]
            ==
            "validated"
        )


        assert (
            contract[
                "request_text"
            ]
            ==
            "Détecte les outliers."
        )


        assert (
            contract[
                "required_dataset_ids"
            ]
            ==
            [
                "dataset:0001"
            ]
        )


        value_bindings = [
            binding

            for binding
            in contract[
                "bindings"
            ]

            if (
                binding[
                    "role"
                ]
                ==
                "value"
            )
        ]


        assert (
            len(
                value_bindings
            )
            ==
            1
        )


        binding = (
            value_bindings[
                0
            ]
        )


        assert (
            binding[
                "analysis_kind"
            ]
            ==
            "quantitative"
        )


        target_columns.add(
            binding[
                "column"
            ]
        )


    print(
        "\n=== RESOLVED TARGETS ==="
    )


    for column_name in sorted(
        target_columns
    ):
        print(
            f"- {column_name}"
        )


    # ========================================================
    # NO HALLUCINATED OR IDENTIFIER TARGET
    # ========================================================

    assert (
        "price"
        not in
        target_columns
    )


    assert (
        "customer_id"
        not in
        target_columns
    )


    expected_numeric_columns = {
        "age",
        "annual_salary",
        "satisfaction_score",
    }


    assert (
        target_columns
        .issubset(
            expected_numeric_columns
        )
    )


    assert (
        target_columns
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:
    print(
        "\n========================================"
    )

    print(
        (
            "DataLens Analysis Intent Routing "
            "API v0.1"
        )
    )

    print(
        "========================================"
    )


    test_generic_outlier_preview_uses_python_router()


    print(
        "\n========================================"
    )

    print(
        (
            "PASS - analysis intent routing "
            "API v0.1"
        )
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()

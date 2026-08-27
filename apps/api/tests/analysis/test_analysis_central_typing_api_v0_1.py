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
#
# This fixture reproduces the semantic shapes that motivated
# the central planner catalog:
#
# - birth     : numeric storage, temporal semantics
# - price     : genuine quantitative measure
# - categ     : numeric category code
# - client_id : identifier
# ============================================================


def build_lapage_like_csv(
) -> str:
    rows = [
        (
            "client_id,"
            "birth,"
            "price,"
            "categ"
        )
    ]


    for index in range(
        300
    ):
        client_id = (
            f"c_{index:04d}"
        )


        birth = (
            1950
            +
            (
                index
                %
                51
            )
        )


        # Deliberately unique continuous values.
        # A unique numerical measure must remain quantitative.
        price = round(
            5.17
            +
            (
                index
                *
                1.379
            ),
            3,
        )


        categ = (
            index
            %
            3
        )


        rows.append(
            (
                f"{client_id},"
                f"{birth},"
                f"{price},"
                f"{categ}"
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
# GEMMA MUST NOT RUN
# ============================================================


def forbidden_ai_fallback(
    *args,
    **kwargs,
):
    del args
    del kwargs


    raise AssertionError(
        (
            "Gemma must not be called for the generic "
            "outlier request."
        )
    )


# ============================================================
# TEST
# ============================================================


def test_generic_outlier_preview_uses_central_analytical_typing(
) -> None:
    csv_content = (
        build_lapage_like_csv()
    )


    workflow_id = (
        prepare_validated_workflow(
            csv_content=
                csv_content,

            dataset_filename=
                "products.csv",
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


    print()
    print(
        "=============================================="
    )

    print(
        "DataLens Analysis Central Typing API v0.1"
    )

    print(
        "=============================================="
    )


    print()
    print(
        f"HTTP status       : {response.status_code}"
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
        f"Planner model     : {body['model']}"
    )

    print(
        f"Proposal count    : {body['proposal_count']}"
    )

    print(
        f"Validated count   : {body['validated_count']}"
    )

    print(
        f"Rejected count    : {body['rejected_count']}"
    )

    print(
        (
            "Model inference   : "
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
    # EXACT GENERIC TARGET
    # ========================================================

    assert (
        body[
            "proposal_count"
        ]
        ==
        1
    )


    assert (
        body[
            "validated_count"
        ]
        ==
        1
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


    item = (
        body[
            "items"
        ][
            0
        ]
    )


    assert (
        item[
            "validation_status"
        ]
        ==
        "validated"
    )


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
            "request_text"
        ]
        ==
        "Détecte les outliers."
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
            "column"
        ]
        ==
        "price"
    )


    assert (
        binding[
            "analysis_kind"
        ]
        ==
        "quantitative"
    )


    # ========================================================
    # SEMANTIC EXCLUSIONS
    # ========================================================

    target_columns = {
        bound[
            "column"
        ]

        for report_item
        in body[
            "items"
        ]

        if (
            report_item[
                "contract"
            ]
            is not None
        )

        for bound
        in report_item[
            "contract"
        ][
            "bindings"
        ]

        if (
            bound[
                "role"
            ]
            ==
            "value"
        )
    }


    assert (
        target_columns
        ==
        {
            "price"
        }
    )


    assert (
        "birth"
        not in
        target_columns
    )


    assert (
        "categ"
        not in
        target_columns
    )


    assert (
        "client_id"
        not in
        target_columns
    )


    print()
    print(
        "=== RESOLVED OUTLIER TARGETS ==="
    )


    for column_name in sorted(
        target_columns
    ):
        print(
            f"- {column_name}"
        )


    print()
    print(
        "birth     -> EXCLUDED (temporal)"
    )

    print(
        "categ     -> EXCLUDED (categorical)"
    )

    print(
        "client_id -> EXCLUDED (identifier)"
    )

    print(
        "price     -> INCLUDED (quantitative)"
    )


    print()
    print(
        "=============================================="
    )

    print(
        "PASS - analysis central typing API v0.1"
    )

    print(
        "=============================================="
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:
    test_generic_outlier_preview_uses_central_analytical_typing()


if __name__ == "__main__":
    main()

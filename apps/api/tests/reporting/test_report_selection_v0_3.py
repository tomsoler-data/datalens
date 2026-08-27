from __future__ import annotations


import os

from pathlib import (
    Path,
)

from tempfile import (
    TemporaryDirectory,
)

from types import (
    SimpleNamespace,
)


from app.reporting.analysis_artifact_store import (
    delete_analysis_artifacts,
    list_analysis_artifact_details,
    register_native_pipeline_result,
)

from app.reporting.report_selection_store import (
    add_analysis_to_report,
    delete_report_selection,
    get_report_selection_details,
    remove_analysis_from_report,
)

from app.reporting.selected_report_pdf import (
    build_selected_report_pdf,
)

from app.reporting.unified_report_artifacts import (
    register_unified_report_artifacts,
)


class FakeNativePipeline:
    def __init__(
        self,
        *,
        trace_id: str,
        objective: str,
    ) -> None:
        self.trace_id = (
            trace_id
        )

        self.planner = (
            SimpleNamespace(
                objective=
                    objective
            )
        )

        self.executed_count = (
            1
        )

        self._payload = {
            "trace_id":
                trace_id,

            "status":
                "ready",

            "planner_model":
                "gemma3:4b",

            "tool_model":
                "qwen2.5:1.5b-instruct",

            "planner":
                {
                    "status":
                        "ready",

                    "objective":
                        objective,

                    "model":
                        "gemma3:4b",

                    "proposal_count":
                        1,

                    "validated_count":
                        1,

                    "blocked_count":
                        0,

                    "ambiguous_count":
                        0,

                    "rejected_count":
                        0,

                    "items":
                        [],

                    "planner_rule_version":
                        "fake",
                },

            "validated_contract_count":
                1,

            "pipeline_item_count":
                1,

            "executed_count":
                1,

            "not_supported_count":
                0,

            "rejected_count":
                0,

            "items":
                [
                    {
                        "contract_id":
                            "contract:fake",

                        "family":
                            "ranking",

                        "pipeline_status":
                            "executed",

                        "native_tool":
                            {
                                "model":
                                    "qwen",

                                "requested_tool":
                                    "run_ranking",

                                "requested_arguments":
                                    {},

                                "validation_status":
                                    "validated",

                                "validation_errors":
                                    [],

                                "attempt_count":
                                    1,

                                "retry_count":
                                    0,

                                "attempts":
                                    [],

                                "tool_call_received":
                                    True,

                                "execution":
                                    {
                                        "tool_name":
                                            "run_ranking",

                                        "execution_status":
                                            "executed",

                                        "dataset_id":
                                            "dataset:final",

                                        "dataset_filename":
                                            "final.csv",

                                        "arguments":
                                            {},

                                        "result":
                                            {
                                                "title":
                                                    objective,

                                                "family":
                                                    "ranking",

                                                "execution_status":
                                                    "complete",

                                                "chart_type":
                                                    "bar",

                                                "summary":
                                                    [
                                                        "Résultat demandé."
                                                    ],

                                                "metrics":
                                                    {
                                                        "result_count":
                                                            2,
                                                    },

                                                "chart_data":
                                                    [
                                                        {
                                                            "category":
                                                                "A",

                                                            "value":
                                                                10.0,
                                                        },
                                                        {
                                                            "category":
                                                                "B",

                                                            "value":
                                                                8.0,
                                                        },
                                                    ],

                                                "warnings":
                                                    [],

                                                "limitations":
                                                    [],
                                            },

                                        "errors":
                                            [],

                                        "warnings":
                                            [],
                                    },

                                "native_tool_rule_version":
                                    "fake",
                            },

                        "errors":
                            [],

                        "warnings":
                            [],
                    }
                ],

            "notes":
                [],

            "pipeline_rule_version":
                "fake",
        }


    def model_dump(
        self,
        *,
        mode: str,
    ):
        _ = mode

        return dict(
            self._payload
        )


def dataset_records(
    workflow_id: str,
):
    return [
        {
            "dataset_id":
                "dataset:final",

            "filename":
                "final.csv",

            "preparation_workflow_id":
                workflow_id,
        }
    ]


def fake_report() -> dict:
    return {
        "requested_findings":
            [
                {
                    "request_id":
                        "request:doc:1",

                    "analysis_id":
                        "requested:doc:1",

                    "request_text":
                        (
                            "Comparer les catégories "
                            "demandées dans le brief."
                        ),

                    "title":
                        "Analyse demandée dans le document",

                    "origin":
                        "requested",

                    "family":
                        "group_comparison",

                    "execution_status":
                        "complete",

                    "dataset_id":
                        "dataset:final",

                    "datasets":
                        [
                            "dataset:final"
                        ],

                    "analytical_grain":
                        "transaction",

                    "variables":
                        {
                            "group":
                                "categ",

                            "value":
                                "price",
                        },

                    "summary":
                        [
                            "Analyse documentaire exécutée."
                        ],

                    "reasons":
                        [],

                    "caveats":
                        [],

                    "chart_type":
                        "bar",

                    "chart_data":
                        [
                            {
                                "category":
                                    "0",

                                "value":
                                    10.0,
                            }
                        ],

                    "metrics":
                        {
                            "valid_observations":
                                100,
                        },

                    "source_filename":
                        "brief.pdf",

                    "source_locator":
                        "page:2",

                    "page_number":
                        2,

                    "evidence_quote":
                        "Comparer les catégories.",

                    "adapter_rule_version":
                        "requested_adapter_v0.1",
                }
            ],

        "main_findings":
            [
                {
                    "analysis_id":
                        "auto:1",

                    "title":
                        "Évolution des ventes",

                    "family":
                        "time_series",

                    "dataset_id":
                        "dataset:final",

                    "datasets":
                        [
                            "dataset:final"
                        ],

                    "analytical_grain":
                        "month",

                    "variables":
                        {
                            "time":
                                "date",

                            "value":
                                "price",
                        },

                    "summary":
                        [
                            "Analyse automatique exécutée."
                        ],

                    "reasons":
                        [],

                    "caveats":
                        [],

                    "chart_type":
                        "bar",

                    "chart_data":
                        [
                            {
                                "category":
                                    "Mars",

                                "value":
                                    12.0,
                            }
                        ],

                    "metrics":
                        {
                            "valid_observations":
                                100,
                        },
                }
            ],

        "additional_findings":
            [
                {
                    "analysis_id":
                        "auto:2",

                    "title":
                        "Distribution des prix",

                    "family":
                        "distribution",

                    "dataset_id":
                        "dataset:final",

                    "datasets":
                        [
                            "dataset:final"
                        ],

                    "variables":
                        {
                            "value":
                                "price",
                        },

                    "summary":
                        [
                            "Distribution automatique."
                        ],

                    "reasons":
                        [],

                    "caveats":
                        [],

                    "chart_type":
                        "none",

                    "chart_data":
                        [],

                    "metrics":
                        {
                            "valid_observations":
                                100,
                        },
                }
            ],

        "context_analyses":
            [],
    }


def main() -> None:
    with TemporaryDirectory() as temporary:
        temp = Path(
            temporary
        )


        os.environ[
            "DATALENS_ANALYSIS_ARTIFACT_STORE_PATH"
        ] = str(
            temp
            /
            "analysis_artifacts.json"
        )


        os.environ[
            "DATALENS_REPORT_SELECTION_STORE_PATH"
        ] = str(
            temp
            /
            "report_selection.json"
        )


        delete_analysis_artifacts()

        delete_report_selection()


        workflow_id = (
            "workflow:selection-v0-3"
        )


        registered = (
            register_unified_report_artifacts(
                workflow_id=
                    workflow_id,

                report=
                    fake_report(),
            )
        )


        assert len(
            registered
        ) == 3


        available = (
            list_analysis_artifact_details(
                workflow_id=
                    workflow_id
            )
        )


        assert available.count == 3


        selected = (
            get_report_selection_details(
                workflow_id=
                    workflow_id
            )
        )


        assert selected.selected_count == 1

        assert (
            selected.analyses[
                0
            ].selection.source_type
            ==
            "document_request"
        )


        print(
            "[PASS] document-requested analysis selected by default"
        )


        automatic = [
            item

            for item
            in available.analyses

            if (
                item.source_type
                ==
                "automatic"
            )
        ]


        assert len(
            automatic
        ) == 2


        selected_ids = {
            item.selection.analysis_id

            for item
            in selected.analyses
        }


        assert all(
            item.analysis_id
            not in selected_ids

            for item
            in automatic
        )


        print(
            "[PASS] automatic analyses remain optional"
        )


        add_analysis_to_report(
            workflow_id=
                workflow_id,

            analysis_id=
                automatic[
                    0
                ].analysis_id,
        )


        after_auto_add = (
            get_report_selection_details(
                workflow_id=
                    workflow_id
            )
        )


        assert (
            automatic[
                0
            ].analysis_id
            in {
                item.selection.analysis_id

                for item
                in after_auto_add.analyses
            }
        )


        print(
            "[PASS] automatic analysis can be explicitly added"
        )


        remove_analysis_from_report(
            workflow_id=
                workflow_id,

            analysis_id=
                automatic[
                    0
                ].analysis_id,
        )


        after_auto_remove = (
            get_report_selection_details(
                workflow_id=
                    workflow_id
            )
        )


        assert (
            automatic[
                0
            ].analysis_id
            not in {
                item.selection.analysis_id

                for item
                in after_auto_remove.analyses
            }
        )


        print(
            "[PASS] automatic analysis can be removed again"
        )


        initial = (
            register_native_pipeline_result(
                datasets=
                    dataset_records(
                        workflow_id
                    ),

                pipeline_report=
                    FakeNativePipeline(
                        trace_id=
                            "trace:prompt:1",

                        objective=
                            "Compare les prix par catégorie.",
                    ),
            )
        )


        assert initial is not None

        assert (
            initial.source_type
            ==
            "initial_request"
        )


        after_initial = (
            get_report_selection_details(
                workflow_id=
                    workflow_id
            )
        )


        assert (
            initial.analysis_id
            in {
                item.selection.analysis_id

                for item
                in after_initial.analyses
            }
        )


        print(
            "[PASS] first prompt stays initial after automatic/document registration"
        )

        print(
            "[PASS] initial prompt selected by default"
        )


        follow_up = (
            register_native_pipeline_result(
                datasets=
                    dataset_records(
                        workflow_id
                    ),

                pipeline_report=
                    FakeNativePipeline(
                        trace_id=
                            "trace:prompt:2",

                        objective=
                            "Donne les deux catégories principales.",
                    ),
            )
        )


        assert follow_up is not None

        assert (
            follow_up.source_type
            ==
            "follow_up_prompt"
        )


        after_follow_up = (
            get_report_selection_details(
                workflow_id=
                    workflow_id
            )
        )


        assert (
            follow_up.analysis_id
            in {
                item.selection.analysis_id

                for item
                in after_follow_up.analyses
            }
        )


        print(
            "[PASS] follow-up prompt selected by default"
        )


        remove_analysis_from_report(
            workflow_id=
                workflow_id,

            analysis_id=
                follow_up.analysis_id,
        )


        # Re-register same native result: user's removal must survive.
        repeated = (
            register_native_pipeline_result(
                datasets=
                    dataset_records(
                        workflow_id
                    ),

                pipeline_report=
                    FakeNativePipeline(
                        trace_id=
                            "trace:prompt:2",

                        objective=
                            "Donne les deux catégories principales.",
                    ),
            )
        )


        assert repeated is not None


        after_repeat = (
            get_report_selection_details(
                workflow_id=
                    workflow_id
            )
        )


        assert (
            follow_up.analysis_id
            not in {
                item.selection.analysis_id

                for item
                in after_repeat.analyses
            }
        )


        print(
            "[PASS] explicit removal survives refresh/re-registration"
        )


        pdf_bytes = (
            build_selected_report_pdf(
                after_repeat,

                preparation_context=
                    {
                        "workflow_id":
                            workflow_id,

                        "available":
                            False,
                    },
            )
        )


        assert pdf_bytes[
            :
            4
        ] == b"%PDF"


        assert len(
            pdf_bytes
        ) > 3000


        print(
            "[PASS] mixed document + prompt selection exports to PDF"
        )


        print()
        print(
            "PASS - report selection v0.3"
        )


if __name__ == "__main__":
    print(
        "=== DATALENS REPORT SELECTION v0.3 ==="
    )
    print()

    main()

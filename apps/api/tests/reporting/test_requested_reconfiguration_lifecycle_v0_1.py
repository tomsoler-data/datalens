from __future__ import annotations

import os

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from fastapi import HTTPException

import app.api.requested_resolution as resolution_api

from app.api.requested_resolution import (
    RequestedAnalysisReconfigurationRequest,
    reconfigure_requested_analysis_http,
)

from app.planning.schemas import (
    RequestedAnalysisPlan,
    RequestedAnalysisResolution,
    RequestedColumnMatch,
)

from app.reporting.analysis_artifact_store import (
    delete_analysis_artifacts,
    get_analysis_artifact,
)

from app.reporting.report_selection_store import (
    add_analysis_to_report,
    delete_report_selection,
    get_report_selection,
    remove_analysis_from_report,
)

from app.reporting.unified_report_artifacts import (
    register_requested_report_finding,
)


def column_match(
    concept: str,
    column: str,
) -> RequestedColumnMatch:
    return RequestedColumnMatch(
        concept=
            concept,

        dataset_id=
            "dataset:transactions",

        dataset_filename=
            "transactions.csv",

        column=
            column,

        analysis_kind=
            (
                "temporal"
                if concept == "time"
                else
                "quantitative"
            ),

        match_score=
            100,

        reasons=[
            "server-owned test binding"
        ],
    )


def ready_plan(
    *,
    granularity: str,
    window: int,
) -> RequestedAnalysisPlan:
    return RequestedAnalysisPlan(
        request_id=
            "request:test-moving-average",

        request_text=
            (
                "Afficher le chiffre d'affaires "
                "avec une moyenne mobile."
            ),

        evidence_quote=
            (
                "Afficher le chiffre d'affaires "
                "avec une moyenne mobile."
            ),

        source_filename=
            "brief.pdf",

        source_locator=
            "page 1",

        page_number=
            1,

        source_chunk_id=
            "chunk:test-moving-average",

        evidence_unit_id=
            1,

        kind=
            "revenue_moving_average",

        status=
            "ready",

        resolution=
            RequestedAnalysisResolution(
                resolution_type=
                    "time_series_parameters",

                time_granularity=
                    granularity,

                moving_average_window=
                    window,
            ),

        target_family=
            "time_series",

        matched_columns=[
            column_match(
                "amount",
                "price",
            ),

            column_match(
                "time",
                "date",
            ),
        ],

        required_dataset_ids=[
            "dataset:transactions"
        ],

        required_dataset_filenames=[
            "transactions.csv"
        ],

        required_operations=[
            (
                f"Aggregate revenue by {granularity}."
            ),
            (
                f"Compute a moving average over {window} periods."
            ),
        ],

        reasons=[
            "Time-series request detected.",
            (
                "The user explicitly selected "
                f"time granularity={granularity} "
                "and moving-average "
                f"window={window}."
            ),
        ],

        blockers=[],
    )


def requested_source_id(
    request_id: str,
) -> str:
    if request_id.startswith(
        "request:"
    ):
        return (
            "requested:"
            +
            request_id[
                len("request:"):
            ]
        )

    return (
        "requested:"
        +
        request_id
    )


class FakeFinding:
    def __init__(
        self,
        *,
        plan:
            RequestedAnalysisPlan,
    ) -> None:
        assert (
            plan.resolution
            is not None
        )

        self.analysis_id = (
            requested_source_id(
                plan.request_id
            )
        )

        self.request_id = (
            plan.request_id
        )

        self.request_text = (
            plan.request_text
        )

        self.granularity = (
            plan
            .resolution
            .time_granularity
        )

        self.window = (
            plan
            .resolution
            .moving_average_window
        )


    def model_dump(
        self,
        *,
        mode: str = "json",
    ):
        _ = mode

        return {
            "request_id":
                self.request_id,

            "analysis_id":
                self.analysis_id,

            "request_text":
                self.request_text,

            "title":
                "Chiffre d'affaires avec moyenne mobile",

            "origin":
                "requested",

            "kind":
                "revenue_moving_average",

            "scope":
                "single_dataset",

            "family":
                "time_series",

            "execution_status":
                "complete",

            "inferential_status":
                "not_applicable",

            "analysis_mode":
                "descriptive",

            "dataset_id":
                "dataset:transactions",

            "dataset_filename":
                "transactions.csv",

            "datasets":
                [
                    "dataset:transactions"
                ],

            "analytical_grain":
                "transaction",

            "variables":
                {
                    "time":
                        "date",

                    "value":
                        "price",
                },

            "sample_size":
                100,

            "summary":
                [
                    "Requested moving average executed."
                ],

            "reasons":
                [],

            "caveats":
                [],

            "chart_type":
                "line",

            "chart_data":
                [
                    {
                        "period":
                            "2026-01-01",

                        "value":
                            100.0,

                        "moving_average":
                            100.0,
                    }
                ],

            "metrics":
                {
                    "valid_observations":
                        100,

                    "period_count":
                        1,

                    "aggregation_period":
                        self.granularity,

                    "moving_average_window":
                        self.window,
                },

            "source_filename":
                "brief.pdf",

            "source_locator":
                "page 1",

            "page_number":
                1,

            "source_chunk_id":
                "chunk:test-moving-average",

            "evidence_unit_id":
                1,

            "evidence_quote":
                (
                    "Afficher le chiffre d'affaires "
                    "avec une moyenne mobile."
                ),

            "adapter_rule_version":
                "requested_reconfiguration_test_v0.1",
        }


def fake_handoff(
    *,
    workflow_id: str,
):
    return SimpleNamespace(
        dataset_records=[
            {
                "dataset_id":
                    "dataset:transactions",

                "filename":
                    "transactions.csv",

                "preparation_workflow_id":
                    workflow_id,
            }
        ]
    )


def fake_prepare_analysis_datasets(
    **kwargs,
):
    _ = kwargs

    return (
        SimpleNamespace(),
        [],
    )


def fake_execute_requested_analysis(
    *,
    request,
    datasets,
):
    _ = datasets

    assert (
        request.status
        ==
        "ready"
    )

    assert (
        request.resolution
        is not None
    )

    return SimpleNamespace(
        request_id=
            request.request_id,

        execution_status=
            "complete",

        warnings=
            [],

        limitations=
            [],
    )


def fake_build_requested_report_finding(
    execution,
    *,
    plan,
):
    assert (
        execution.request_id
        ==
        plan.request_id
    )

    return (
        FakeFinding(
            plan=
                plan
        )
    )


def selected_ids(
    workflow_id: str,
) -> set[str]:
    selection = (
        get_report_selection(
            workflow_id=
                workflow_id
        )
    )

    return {
        item.analysis_id
        for item
        in selection.analyses
    }


def main() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(
            temporary
        )


        os.environ[
            "DATALENS_ANALYSIS_ARTIFACT_STORE_PATH"
        ] = str(
            root
            /
            "analysis_artifacts.json"
        )

        os.environ[
            "DATALENS_REPORT_SELECTION_STORE_PATH"
        ] = str(
            root
            /
            "report_selection.json"
        )


        delete_analysis_artifacts()

        delete_report_selection()


        workflow_id = (
            "workflow:requested-reconfiguration"
        )


        original_plan = (
            ready_plan(
                granularity=
                    "week",

                window=
                    4,
            )
        )


        before = (
            register_requested_report_finding(
                workflow_id=
                    workflow_id,

                finding=
                    FakeFinding(
                        plan=
                            original_plan
                    ),

                requested_plan=
                    original_plan,

                select_by_default=
                    True,
            )
        )


        # Report composition is manual-only.
        #
        # Registration must never select an analysis merely
        # because an older call site passes
        # select_by_default=True.
        #
        # Explicitly model the user's selection so this test
        # can verify that reconfiguration preserves it.
        add_analysis_to_report(
            workflow_id=
                workflow_id,

            analysis_id=
                before.analysis_id,
        )


        assert (
            before.analysis_id
            in
            selected_ids(
                workflow_id
            )
        )


        original_created_at = (
            before.created_at_utc
        )


        original_handoff = (
            resolution_api
            .load_validated_analysis_input_for_http
        )

        original_prepare = (
            resolution_api
            .prepare_analysis_datasets
        )

        original_execute = (
            resolution_api
            .execute_requested_analysis
        )

        original_adapter = (
            resolution_api
            .build_requested_report_finding
        )


        resolution_api.load_validated_analysis_input_for_http = (
            fake_handoff
        )

        resolution_api.prepare_analysis_datasets = (
            fake_prepare_analysis_datasets
        )

        resolution_api.execute_requested_analysis = (
            fake_execute_requested_analysis
        )

        resolution_api.build_requested_report_finding = (
            fake_build_requested_report_finding
        )


        try:
            print(
                "=== DATALENS REQUESTED RECONFIGURATION LIFECYCLE v0.1 ==="
            )
            print()


            # ====================================================
            # 1. WEEK / 4 -> MONTH / 3
            # ====================================================

            response = (
                reconfigure_requested_analysis_http(
                    RequestedAnalysisReconfigurationRequest(
                        workflow_id=
                            workflow_id,

                        request_id=
                            original_plan.request_id,

                        resolution=
                            RequestedAnalysisResolution(
                                resolution_type=
                                    "time_series_parameters",

                                time_granularity=
                                    "month",

                                moving_average_window=
                                    3,
                            ),
                    )
                )
            )


            assert (
                response.plan_status
                ==
                "ready"
            )

            assert (
                response.execution_status
                ==
                "complete"
            )

            assert (
                response.executed
            )

            assert (
                response.analysis_id
                ==
                before.analysis_id
            )

            assert (
                response.resolution.time_granularity
                ==
                "month"
            )

            assert (
                response.resolution.moving_average_window
                ==
                3
            )


            after = (
                get_analysis_artifact(
                    workflow_id=
                        workflow_id,

                    analysis_id=
                        before.analysis_id,
                )
            )


            assert (
                after.analysis_id
                ==
                before.analysis_id
            )

            assert (
                after.created_at_utc
                ==
                original_created_at
            )

            assert (
                after.source_type
                ==
                "document_request"
            )

            assert (
                after.executed
            )


            stored_plan = (
                after
                .pipeline_payload[
                    "requested_plan"
                ]
            )

            stored_finding = (
                after
                .pipeline_payload[
                    "requested_finding"
                ]
            )


            assert (
                stored_plan[
                    "resolution"
                ][
                    "time_granularity"
                ]
                ==
                "month"
            )

            assert (
                stored_plan[
                    "resolution"
                ][
                    "moving_average_window"
                ]
                ==
                3
            )

            assert (
                stored_finding[
                    "metrics"
                ][
                    "aggregation_period"
                ]
                ==
                "month"
            )

            assert (
                stored_finding[
                    "metrics"
                ][
                    "moving_average_window"
                ]
                ==
                3
            )


            print(
                "[PASS] week / 4 reconfigures to month / 3"
            )

            print(
                "[PASS] request and artifact identity are preserved"
            )

            print(
                "[PASS] artifact created_at is preserved"
            )

            print(
                "[PASS] requested plan and finding refresh together"
            )


            # ====================================================
            # 2. SELECTED REMAINS SELECTED
            # ====================================================

            assert (
                before.analysis_id
                in
                selected_ids(
                    workflow_id
                )
            )


            print(
                "[PASS] selected analysis remains selected"
            )


            # ====================================================
            # 3. USER DESELECTS THEN RECONFIGURES AGAIN
            # ====================================================

            remove_analysis_from_report(
                workflow_id=
                    workflow_id,

                analysis_id=
                    before.analysis_id,
            )


            assert (
                before.analysis_id
                not in
                selected_ids(
                    workflow_id
                )
            )


            second_response = (
                reconfigure_requested_analysis_http(
                    RequestedAnalysisReconfigurationRequest(
                        workflow_id=
                            workflow_id,

                        request_id=
                            original_plan.request_id,

                        resolution=
                            RequestedAnalysisResolution(
                                resolution_type=
                                    "time_series_parameters",

                                time_granularity=
                                    "quarter",

                                moving_average_window=
                                    2,
                            ),
                    )
                )
            )


            assert (
                second_response
                .resolution
                .time_granularity
                ==
                "quarter"
            )

            assert (
                second_response
                .resolution
                .moving_average_window
                ==
                2
            )


            assert (
                before.analysis_id
                not in
                selected_ids(
                    workflow_id
                )
            )


            second_after = (
                get_analysis_artifact(
                    workflow_id=
                        workflow_id,

                    analysis_id=
                        before.analysis_id,
                )
            )


            assert (
                second_after.created_at_utc
                ==
                original_created_at
            )


            assert (
                second_after
                .pipeline_payload[
                    "requested_plan"
                ][
                    "resolution"
                ][
                    "time_granularity"
                ]
                ==
                "quarter"
            )


            print(
                "[PASS] deselected analysis remains deselected"
            )

            print(
                "[PASS] repeated reconfiguration refreshes same artifact"
            )


            # ====================================================
            # 4. /resolve SEMANTICS REMAIN DISTINCT
            # ====================================================

            paths = {
                route.path
                for route
                in resolution_api.router.routes
            }


            assert (
                "/analysis/requested/resolve"
                in
                paths
            )

            assert (
                "/analysis/requested/reconfigure"
                in
                paths
            )


            print(
                "[PASS] resolve and reconfigure are separate HTTP operations"
            )


            # ====================================================
            # 5. NON TIME-SERIES RESOLUTION MUST FAIL CLOSED
            # ====================================================

            rejected = False


            try:
                reconfigure_requested_analysis_http(
                    RequestedAnalysisReconfigurationRequest(
                        workflow_id=
                            workflow_id,

                        request_id=
                            original_plan.request_id,

                        resolution=
                            RequestedAnalysisResolution(
                                resolution_type=
                                    "ranking_metric",

                                ranking_metric=
                                    "revenue",
                            ),
                    )
                )

            except HTTPException as error:
                rejected = True

                assert (
                    error.status_code
                    ==
                    409
                )

                assert (
                    error.detail[
                        "error"
                    ]
                    ==
                    "requested_reconfiguration_rejected"
                )


            assert (
                rejected
            )


            print(
                "[PASS] ranking resolution cannot reconfigure time series"
            )


            print()
            print(
                "PASS - requested reconfiguration lifecycle v0.1"
            )


        finally:
            resolution_api.load_validated_analysis_input_for_http = (
                original_handoff
            )

            resolution_api.prepare_analysis_datasets = (
                original_prepare
            )

            resolution_api.execute_requested_analysis = (
                original_execute
            )

            resolution_api.build_requested_report_finding = (
                original_adapter
            )


if __name__ == "__main__":
    main()

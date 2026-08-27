from __future__ import annotations

import os

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from fastapi import HTTPException

import app.api.requested_resolution as resolution_api

from app.api.requested_resolution import (
    RequestedAnalysisResolutionRequest,
    resolve_requested_analysis_http,
)

from app.planning.schemas import (
    RequestedAnalysisPlan,
    RequestedAnalysisResolution,
    RequestedColumnMatch,
)

from app.reporting.analysis_artifact_store import (
    delete_analysis_artifacts,
    get_analysis_artifact,
    register_server_owned_analysis,
)

from app.reporting.report_selection_store import (
    delete_report_selection,
    get_report_selection,
)

from app.reporting.unified_report_artifacts import (
    _stable_id,
)


# ============================================================
# TEST HELPERS
# ============================================================

def column_match(
    concept: str,
    column: str,
) -> RequestedColumnMatch:
    return RequestedColumnMatch(
        concept=concept,
        dataset_id="dataset:transactions",
        dataset_filename="transactions.csv",
        column=column,
        analysis_kind="ranking",
        match_score=100,
        reasons=[
            "test"
        ],
    )


def ambiguous_plan(
    *,
    request_id: str,
) -> RequestedAnalysisPlan:
    return RequestedAnalysisPlan(
        request_id=
            request_id,

        request_text=
            "les tops",

        evidence_quote=
            "les tops",

        source_filename=
            "brief.pdf",

        source_locator=
            "page 1",

        page_number=
            1,

        source_chunk_id=
            "chunk:test",

        evidence_unit_id=
            6,

        kind=
            "top_products",

        status=
            "ambiguous",

        target_family=
            "ranking",

        matched_columns=[
            column_match(
                "product_id",
                "id_prod",
            ),
            column_match(
                "amount",
                "price",
            ),
            column_match(
                "session_id",
                "session_id",
            ),
        ],

        required_dataset_ids=[
            "dataset:transactions"
        ],

        required_dataset_filenames=[
            "transactions.csv"
        ],

        required_operations=[
            "Resolve ranking metric."
        ],

        reasons=[
            "Ranking intent detected."
        ],

        blockers=[
            "Ranking metric is ambiguous."
        ],
    )


def ambiguous_time_series_plan(
    *,
    request_id: str,
) -> RequestedAnalysisPlan:
    return RequestedAnalysisPlan(
        request_id=
            request_id,

        request_text=
            (
                "chiffre d'affaires avec moyenne "
                "mobile, choix jour semaine mois"
            ),

        evidence_quote=
            (
                "chiffre d'affaires avec moyenne "
                "mobile"
            ),

        source_filename=
            "brief.pdf",

        source_locator=
            "page 1",

        page_number=
            1,

        source_chunk_id=
            "chunk:time-series",

        evidence_unit_id=
            1,

        kind=
            "revenue_moving_average",

        status=
            "ambiguous",

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
                "Resolve time-series "
                "parameters."
            )
        ],

        reasons=[
            (
                "The user may choose the "
                "time granularity."
            )
        ],

        blockers=[
            (
                "Time granularity and "
                "moving-average window "
                "require user resolution."
            )
        ],
    )


def model_dump(
    model,
):
    if hasattr(
        model,
        "model_dump",
    ):
        return model.model_dump(
            mode="json"
        )

    return model.dict()


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
                len(
                    "request:"
                ):
            ]
        )

    return (
        "requested:"
        +
        request_id
    )


def create_lifecycle_artifact(
    *,
    workflow_id: str,
    request_id: str,
):
    plan = (
        ambiguous_plan(
            request_id=
                request_id
        )
    )

    source_analysis_id = (
        requested_source_id(
            request_id
        )
    )

    analysis_id = (
        _stable_id(
            workflow_id=
                workflow_id,

            source_type=
                "document_request",

            source_analysis_id=
                source_analysis_id,
        )
    )

    artifact = (
        register_server_owned_analysis(
            workflow_id=
                workflow_id,

            analysis_id=
                analysis_id,

            trace_id=
                (
                    "report:"
                    +
                    source_analysis_id
                ),

            source_type=
                "document_request",

            objective=
                plan.request_text,

            executed=
                False,

            executed_count=
                0,

            pipeline_payload={
                "artifact_kind":
                    "requested_analysis_lifecycle",

                "status":
                    "not_executed",

                "request_lifecycle":
                    {
                        "request_id":
                            request_id,

                        "request_text":
                            plan.request_text,

                        "plan_status":
                            "ambiguous",

                        "execution_status":
                            "not_executed",
                    },

                "requested_plan":
                    model_dump(
                        plan
                    ),
            },

            select_by_default=
                False,
        )
    )

    return (
        plan,
        artifact,
    )


def create_time_series_lifecycle_artifact(
    *,
    workflow_id: str,
    request_id: str,
):
    plan = (
        ambiguous_time_series_plan(
            request_id=
                request_id
        )
    )


    source_analysis_id = (
        requested_source_id(
            request_id
        )
    )


    analysis_id = (
        _stable_id(
            workflow_id=
                workflow_id,

            source_type=
                "document_request",

            source_analysis_id=
                source_analysis_id,
        )
    )


    artifact = (
        register_server_owned_analysis(
            workflow_id=
                workflow_id,

            analysis_id=
                analysis_id,

            trace_id=
                (
                    "report:"
                    +
                    source_analysis_id
                ),

            source_type=
                "document_request",

            objective=
                plan.request_text,

            executed=
                False,

            executed_count=
                0,

            pipeline_payload={
                "artifact_kind":
                    "requested_analysis_lifecycle",

                "status":
                    "not_executed",

                "request_lifecycle":
                    {
                        "request_id":
                            request_id,

                        "request_text":
                            plan.request_text,

                        "plan_status":
                            "ambiguous",

                        "execution_status":
                            "not_executed",
                    },

                "requested_plan":
                    model_dump(
                        plan
                    ),
            },

            select_by_default=
                False,
        )
    )


    return (
        plan,
        artifact,
    )


class FakeFinding:
    def __init__(
        self,
        *,
        analysis_id: str,
        request_id: str,
        request_text: str,
    ) -> None:
        self.analysis_id = (
            analysis_id
        )

        self.request_id = (
            request_id
        )

        self.request_text = (
            request_text
        )


    def model_dump(
        self,
        *,
        mode: str = "json",
    ):
        _ = mode

        return {
            "analysis_id":
                self.analysis_id,

            "request_id":
                self.request_id,

            "request_text":
                self.request_text,

            "title":
                self.request_text,

            "family":
                "ranking",

            "execution_status":
                "complete",

            "inferential_status":
                "not_selected",

            "dataset_id":
                "dataset:transactions",

            "dataset_filename":
                "transactions.csv",

            "datasets":
                [
                    "dataset:transactions"
                ],

            "analytical_grain":
                "product",

            "variables":
                {
                    "product":
                        "id_prod",

                    "value":
                        "price",
                },

            "summary":
                [
                    "Requested ranking executed."
                ],

            "reasons":
                [
                    "User resolved ranking metric."
                ],

            "caveats":
                [],

            "chart_type":
                "bar",

            "chart_data":
                [
                    {
                        "category":
                            "A",

                        "value":
                            100.0,

                        "rank":
                            1,
                    }
                ],

            "metrics":
                {
                    "ranking_metric":
                        "revenue",

                    "valid_observations":
                        10,
                },

            "source_filename":
                "brief.pdf",

            "source_locator":
                "page 1",

            "page_number":
                1,

            "evidence_quote":
                "les tops",

            "origin":
                "requested",

            "adapter_rule_version":
                "requested_resolution_test_v0.1",
        }


# ============================================================
# FAKE SERVER-OWNED ANALYTICAL DEPENDENCIES
# ============================================================

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


    if (
        request.kind
        ==
        "revenue_moving_average"
    ):
        assert (
            request
            .resolution
            .resolution_type
            ==
            "time_series_parameters"
        )

        assert (
            request
            .resolution
            .time_granularity
            ==
            "week"
        )

        assert (
            request
            .resolution
            .moving_average_window
            ==
            4
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

    return FakeFinding(
        analysis_id=
            requested_source_id(
                plan.request_id
            ),

        request_id=
            plan.request_id,

        request_text=
            plan.request_text,
    )


# ============================================================
# MAIN
# ============================================================

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
                "===== DATALENS REQUESTED RESOLUTION LIFECYCLE v0.1 ====="
            )
            print()


            # ====================================================
            # 1. AMBIGUOUS LIFECYCLE ARTIFACT
            # ====================================================

            workflow_id = (
                "workflow:requested-resolution"
            )

            request_id = (
                "request:test-top"
            )


            (
                _,
                before,
            ) = (
                create_lifecycle_artifact(
                    workflow_id=
                        workflow_id,

                    request_id=
                        request_id,
                )
            )


            assert not (
                before.executed
            )

            assert (
                before.executed_count
                ==
                0
            )

            assert (
                before.source_type
                ==
                "document_request"
            )


            before_selection = (
                get_report_selection(
                    workflow_id=
                        workflow_id
                )
            )


            assert (
                before_selection.selected_count
                ==
                0
            )


            print(
                "[PASS] ambiguous lifecycle starts executed=False"
            )

            print(
                "[PASS] unresolved request is not report-selected"
            )


            # ====================================================
            # 2. RESOLVE WITH REVENUE
            # ====================================================

            response = (
                resolve_requested_analysis_http(
                    RequestedAnalysisResolutionRequest(
                        workflow_id=
                            workflow_id,

                        request_id=
                            request_id,

                        resolution=
                            RequestedAnalysisResolution(
                                ranking_metric=
                                    "revenue"
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
                before.created_at_utc
            )

            assert (
                after.source_type
                ==
                "document_request"
            )

            assert (
                after.executed
            )

            assert (
                after.executed_count
                ==
                1
            )


            print(
                "[PASS] clarification resolves ambiguous -> ready"
            )

            print(
                "[PASS] lifecycle transitions False -> True"
            )

            print(
                "[PASS] analysis_id is preserved"
            )

            print(
                "[PASS] artifact creation identity is preserved"
            )


            # ====================================================
            # 3. FALSE -> TRUE DEFAULT REPORT SELECTION
            # ====================================================

            after_selection = (
                get_report_selection(
                    workflow_id=
                        workflow_id
                )
            )


            selected_ids = {
                item.analysis_id

                for item
                in after_selection.analyses
            }


            assert (
                after_selection.selected_count
                ==
                1
            )

            assert (
                before.analysis_id
                in
                selected_ids
            )


            print(
                "[PASS] newly executable document request is selected by default"
            )


            # ====================================================
            # 4. TIME-SERIES RESOLUTION USES SAME LIFECYCLE
            # ====================================================

            time_workflow_id = (
                "workflow:requested-resolution-time-series"
            )

            time_request_id = (
                "request:test-moving-average"
            )


            (
                _,
                time_before,
            ) = (
                create_time_series_lifecycle_artifact(
                    workflow_id=
                        time_workflow_id,

                    request_id=
                        time_request_id,
                )
            )


            assert not (
                time_before.executed
            )

            assert (
                time_before.executed_count
                ==
                0
            )


            time_before_selection = (
                get_report_selection(
                    workflow_id=
                        time_workflow_id
                )
            )


            assert (
                time_before_selection.selected_count
                ==
                0
            )


            time_response = (
                resolve_requested_analysis_http(
                    RequestedAnalysisResolutionRequest(
                        workflow_id=
                            time_workflow_id,

                        request_id=
                            time_request_id,

                        resolution=
                            RequestedAnalysisResolution(
                                resolution_type=
                                    "time_series_parameters",

                                time_granularity=
                                    "week",

                                moving_average_window=
                                    4,
                            ),
                    )
                )
            )


            assert (
                time_response.plan_status
                ==
                "ready"
            )

            assert (
                time_response.execution_status
                ==
                "complete"
            )

            assert (
                time_response.executed
            )

            assert (
                time_response.analysis_id
                ==
                time_before.analysis_id
            )

            assert (
                time_response
                .resolution
                .resolution_type
                ==
                "time_series_parameters"
            )

            assert (
                time_response
                .resolution
                .time_granularity
                ==
                "week"
            )

            assert (
                time_response
                .resolution
                .moving_average_window
                ==
                4
            )


            time_after = (
                get_analysis_artifact(
                    workflow_id=
                        time_workflow_id,

                    analysis_id=
                        time_before.analysis_id,
                )
            )


            assert (
                time_after.analysis_id
                ==
                time_before.analysis_id
            )

            assert (
                time_after.created_at_utc
                ==
                time_before.created_at_utc
            )

            assert (
                time_after.source_type
                ==
                "document_request"
            )

            assert (
                time_after.executed
            )

            assert (
                time_after.executed_count
                ==
                1
            )


            time_selection = (
                get_report_selection(
                    workflow_id=
                        time_workflow_id
                )
            )


            time_selected_ids = {
                item.analysis_id

                for item
                in time_selection.analyses
            }


            assert (
                time_selection.selected_count
                ==
                1
            )

            assert (
                time_before.analysis_id
                in
                time_selected_ids
            )


            print(
                "[PASS] time-series clarification resolves ambiguous -> ready"
            )

            print(
                "[PASS] week + window=4 cross the HTTP trust boundary"
            )

            print(
                "[PASS] time-series lifecycle transitions False -> True"
            )

            print(
                "[PASS] time-series analysis_id is preserved"
            )

            print(
                "[PASS] time-series artifact keeps original created_at"
            )

            print(
                "[PASS] newly executable time-series request is selected by default"
            )


            # ====================================================
            # 5. SECOND RESOLUTION MUST FAIL CLOSED
            # ====================================================

            repeated_failed = False


            try:
                resolve_requested_analysis_http(
                    RequestedAnalysisResolutionRequest(
                        workflow_id=
                            workflow_id,

                        request_id=
                            request_id,

                        resolution=
                            RequestedAnalysisResolution(
                                ranking_metric=
                                    "revenue"
                            ),
                    )
                )


            except HTTPException as error:
                repeated_failed = True

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
                    "requested_analysis_already_executed"
                )


            assert (
                repeated_failed
            )


            print(
                "[PASS] already executed request cannot be resolved again"
            )


            # ====================================================
            # 6. NON-EXECUTABLE RESOLUTION MUST NOT PROMOTE
            # ====================================================

            blocked_workflow_id = (
                "workflow:requested-resolution-units"
            )

            blocked_request_id = (
                "request:test-units"
            )


            (
                _,
                blocked_before,
            ) = (
                create_lifecycle_artifact(
                    workflow_id=
                        blocked_workflow_id,

                    request_id=
                        blocked_request_id,
                )
            )


            blocked_failed = False


            try:
                resolve_requested_analysis_http(
                    RequestedAnalysisResolutionRequest(
                        workflow_id=
                            blocked_workflow_id,

                        request_id=
                            blocked_request_id,

                        resolution=
                            RequestedAnalysisResolution(
                                ranking_metric=
                                    "units"
                            ),
                    )
                )


            except HTTPException as error:
                blocked_failed = True

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
                    "requested_resolution_not_executable"
                )


            assert (
                blocked_failed
            )


            blocked_after = (
                get_analysis_artifact(
                    workflow_id=
                        blocked_workflow_id,

                    analysis_id=
                        blocked_before.analysis_id,
                )
            )


            assert not (
                blocked_after.executed
            )

            assert (
                blocked_after.analysis_id
                ==
                blocked_before.analysis_id
            )


            blocked_selection = (
                get_report_selection(
                    workflow_id=
                        blocked_workflow_id
                )
            )


            assert (
                blocked_selection.selected_count
                ==
                0
            )


            print(
                "[PASS] unavailable metric remains fail-closed"
            )

            print(
                "[PASS] failed clarification does not promote artifact"
            )

            print(
                "[PASS] failed clarification does not enter report selection"
            )


            print()
            print(
                "PASS - requested resolution lifecycle v0.1"
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

from __future__ import annotations


import os

from pathlib import Path

from tempfile import (
    TemporaryDirectory,
)

from types import (
    SimpleNamespace,
)


import pandas as pd


import app.api.requested_resolution as resolution_api


from app.api.requested_resolution import (
    FollowUpRequestedAnalysisRouteRequest,
    RequestedAnalysisReconfigurationRequest,
    RequestedAnalysisResolutionRequest,
    reconfigure_requested_analysis_http,
    resolve_requested_analysis_http,
    route_follow_up_requested_analysis_http,
)

from app.ingestion.loader import (
    build_dataset_manifest,
)

from app.ingestion.schemas import (
    MultiDatasetIngestion,
)

from app.planning.follow_up_request import (
    FOLLOW_UP_REQUEST_RULE_VERSION,
    plan_follow_up_requested_analysis,
)

from app.planning.schemas import (
    RequestedAnalysisPlan,
    RequestedAnalysisResolution,
)

from app.reporting.analysis_artifact_store import (
    delete_analysis_artifacts,
    get_analysis_artifact,
    list_analysis_artifacts,
)


# ============================================================
# TEST HELPERS
# ============================================================

def assert_true(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise AssertionError(
            message
        )


def assert_equal(
    actual,
    expected,
    message: str,
) -> None:
    if actual != expected:
        raise AssertionError(
            (
                f"{message} "
                f"expected={expected!r} "
                f"actual={actual!r}"
            )
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


def build_fixture():
    dataframe = pd.DataFrame(
        {
            "transaction_id":
                [
                    "t1",
                    "t2",
                    "t3",
                    "t4",
                ],

            "date":
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-02-01",
                    "2026-02-02",
                ],

            "price":
                [
                    10.0,
                    20.0,
                    30.0,
                    40.0,
                ],
        }
    )


    manifest = (
        build_dataset_manifest(
            dataframe,

            dataset_id=
                "dataset:transactions",

            filename=
                "transactions.csv",

            extension=
                ".csv",
        )
    )


    ingestion = (
        MultiDatasetIngestion(
            dataset_count=
                1,

            total_rows=
                len(
                    dataframe
                ),

            datasets=[
                manifest
            ],

            warnings=[],
        )
    )


    dataset_record = {
        "dataset_id":
            "dataset:transactions",

        "filename":
            "transactions.csv",

        "extension":
            ".csv",

        "dataframe":
            dataframe,

        "preparation_workflow_id":
            "workflow:follow-up-test",
    }


    return (
        ingestion,
        dataset_record,
    )


def fake_prepare_analysis_datasets(
    **kwargs,
):
    source_datasets = list(
        kwargs.get(
            "source_datasets",
            [],
        )
    )

    return (
        SimpleNamespace(),
        source_datasets,
    )


def fake_execute_requested_analysis(
    *,
    request,
    datasets,
):
    _ = datasets

    assert_equal(
        request.status,
        "ready",
        "Resolved request must be ready.",
    )

    return (
        SimpleNamespace(
            request_id=
                request.request_id,

            execution_status=
                "complete",

            warnings=
                [],

            limitations=
                [],
        )
    )


class FakeFinding:
    def __init__(
        self,
        *,
        plan: RequestedAnalysisPlan,
    ) -> None:
        assert_true(
            plan.resolution
            is not None,
            "Resolved plan must contain parameters.",
        )

        self.plan = (
            plan
        )

        self.request_id = (
            plan.request_id
        )

        self.analysis_id = (
            requested_source_id(
                plan.request_id
            )
        )


    def model_dump(
        self,
        *,
        mode: str = "json",
    ):
        _ = mode

        plan = (
            self.plan
        )

        resolution = (
            plan.resolution
        )

        assert resolution is not None


        match_lookup = {
            match.concept:
                match

            for match
            in plan.matched_columns
        }


        time_match = (
            match_lookup.get(
                "time"
            )
        )

        amount_match = (
            match_lookup.get(
                "amount"
            )
        )


        return {
            "request_id":
                plan.request_id,

            "analysis_id":
                self.analysis_id,

            "request_text":
                plan.request_text,

            "title":
                (
                    "Chiffre d'affaires "
                    "avec moyenne mobile"
                ),

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
                (
                    amount_match.dataset_id
                    if amount_match
                    is not None
                    else None
                ),

            "dataset_filename":
                (
                    amount_match.dataset_filename
                    if amount_match
                    is not None
                    else None
                ),

            "datasets":
                list(
                    plan.required_dataset_ids
                ),

            "analytical_grain":
                resolution.time_granularity,

            "variables":
                {
                    "time":
                        (
                            time_match.column
                            if time_match
                            is not None
                            else ""
                        ),

                    "value":
                        (
                            amount_match.column
                            if amount_match
                            is not None
                            else ""
                        ),
                },

            "summary":
                [
                    (
                        "Requested moving-average "
                        "analysis executed."
                    )
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
                            10.0,

                        "moving_average":
                            10.0,
                    }
                ],

            "metrics":
                {
                    "valid_observations":
                        4,

                    "period_count":
                        1,

                    "aggregation_period":
                        resolution.time_granularity,

                    "moving_average_window":
                        resolution.moving_average_window,

                    "moving_average_window_unit":
                        "periods",

                    "moving_average_granularity":
                        resolution.time_granularity,
                },

            "source_filename":
                plan.source_filename,

            "source_locator":
                plan.source_locator,

            "page_number":
                plan.page_number,

            "source_chunk_id":
                plan.source_chunk_id,

            "evidence_unit_id":
                plan.evidence_unit_id,

            "evidence_quote":
                plan.evidence_quote,

            "adapter_rule_version":
                "follow_up_requested_time_series_test_v0.1",
        }


def fake_build_requested_report_finding(
    execution,
    *,
    plan,
):
    assert_equal(
        execution.request_id,
        plan.request_id,
        "Execution/request plan identity mismatch.",
    )

    return (
        FakeFinding(
            plan=
                plan
        )
    )


# ============================================================
# TEST RUNNER
# ============================================================

def main() -> None:
    print(
        "=== DATALENS FOLLOW-UP REQUESTED TIME SERIES v0.1 ==="
    )


    assert_equal(
        FOLLOW_UP_REQUEST_RULE_VERSION,
        "follow_up_requested_analysis_v0.1",
        "Unexpected follow-up planner rule version.",
    )

    print(
        "[PASS] Follow-up planner rule version"
    )


    (
        ingestion,
        dataset_record,
    ) = build_fixture()


    # --------------------------------------------------------
    # 1. Deterministic known-intent classification.
    # --------------------------------------------------------

    report = (
        plan_follow_up_requested_analysis(
            ingestion=
                ingestion,

            objective=
                (
                    "?volution du chiffre d?affaires "
                    "/ moyenne mobile."
                ),

            request_key=
                "planner-test",
        )
    )


    assert_true(
        report
        is not None,
        "Moving-average follow-up was not deterministically routed.",
    )

    assert report is not None

    assert_equal(
        report.request_count,
        1,
        "Expected one deterministic requested plan.",
    )

    assert_equal(
        report.requests[
            0
        ].kind,
        "revenue_moving_average",
        "Unexpected requested analysis kind.",
    )

    assert_equal(
        report.requests[
            0
        ].status,
        "ambiguous",
        (
            "Moving-average follow-up must require explicit "
            "time-series parameters."
        ),
    )

    print(
        "[PASS] Moving-average prompt routes deterministically"
    )


    # --------------------------------------------------------
    # 2. Unsupported prompt leaves deterministic bridge.
    # --------------------------------------------------------

    fallback = (
        plan_follow_up_requested_analysis(
            ingestion=
                ingestion,

            objective=
                "Explique-moi la relation entre deux variables.",

            request_key=
                "fallback-test",
        )
    )


    assert_true(
        fallback
        is None,
        "Unsupported follow-up must fall back to AI-native.",
    )

    print(
        "[PASS] Unsupported prompt preserves AI-native fallback"
    )


    # --------------------------------------------------------
    # 3. HTTP route + lifecycle + resolution + reconfigure.
    # --------------------------------------------------------

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


        delete_analysis_artifacts()


        workflow_id = (
            "workflow:follow-up-test"
        )


        handoff = (
            SimpleNamespace(
                ingestion=
                    ingestion,

                dataset_records=
                    (
                        {
                            **dataset_record,

                            "preparation_workflow_id":
                                workflow_id,
                        },
                    ),
            )
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


        try:
            resolution_api.load_validated_analysis_input_for_http = (
                lambda *,
                workflow_id:
                    handoff
            )

            resolution_api.prepare_analysis_datasets = (
                fake_prepare_analysis_datasets
            )


            routed = (
                route_follow_up_requested_analysis_http(
                    FollowUpRequestedAnalysisRouteRequest(
                        workflow_id=
                            workflow_id,

                        objective=
                            (
                                "?volution du chiffre d?affaires "
                                "/ moyenne mobile."
                            ),
                    )
                )
            )


            assert_equal(
                routed.route_kind,
                "requested_analysis",
                "Known prompt did not use Requested Analysis route.",
            )

            assert_equal(
                routed.kind,
                "revenue_moving_average",
                "HTTP route returned wrong kind.",
            )

            assert_equal(
                routed.plan_status,
                "ambiguous",
                "HTTP route must preserve ambiguity.",
            )

            assert_equal(
                routed.source_type,
                "follow_up_prompt",
                "HTTP route used wrong artifact source type.",
            )

            assert_true(
                bool(
                    routed.analysis_id
                ),
                "HTTP route did not expose lifecycle analysis_id.",
            )

            assert_true(
                bool(
                    routed.request_id
                ),
                "HTTP route did not expose request_id.",
            )


            analysis_id = str(
                routed.analysis_id
            )

            request_id = str(
                routed.request_id
            )


            lifecycle = (
                get_analysis_artifact(
                    workflow_id=
                        workflow_id,

                    analysis_id=
                        analysis_id,
                )
            )


            assert_equal(
                lifecycle.source_type,
                "follow_up_prompt",
                "Lifecycle source type was not preserved.",
            )

            assert_true(
                not lifecycle.executed,
                "Ambiguous lifecycle must remain non-executed.",
            )

            assert_equal(
                lifecycle.pipeline_payload.get(
                    "artifact_kind"
                ),
                "requested_analysis_lifecycle",
                "Missing requested lifecycle payload.",
            )

            print(
                "[PASS] Follow-up lifecycle persisted server-side"
            )


            before_count = len(
                list_analysis_artifacts(
                    workflow_id=
                        workflow_id
                )
            )


            fallback_route = (
                route_follow_up_requested_analysis_http(
                    FollowUpRequestedAnalysisRouteRequest(
                        workflow_id=
                            workflow_id,

                        objective=
                            "Compare deux variables quantitatives.",
                    )
                )
            )


            after_count = len(
                list_analysis_artifacts(
                    workflow_id=
                        workflow_id
                )
            )


            assert_equal(
                fallback_route.route_kind,
                "ai_native",
                "Unknown HTTP prompt must fall back to AI-native.",
            )

            assert_equal(
                after_count,
                before_count,
                "AI-native fallback route must not persist a lifecycle artifact.",
            )

            print(
                "[PASS] HTTP fallback creates no deterministic artifact"
            )


            resolution_api.execute_requested_analysis = (
                fake_execute_requested_analysis
            )

            resolution_api.build_requested_report_finding = (
                fake_build_requested_report_finding
            )


            resolved = (
                resolve_requested_analysis_http(
                    RequestedAnalysisResolutionRequest(
                        workflow_id=
                            workflow_id,

                        request_id=
                            request_id,

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


            assert_equal(
                resolved.analysis_id,
                analysis_id,
                "Resolution changed lifecycle artifact identity.",
            )

            assert_equal(
                resolved.source_type,
                "follow_up_prompt",
                "Resolution changed follow-up source type.",
            )

            assert_true(
                resolved.executed,
                "Resolved follow-up was not promoted to executed.",
            )


            executed_artifact = (
                get_analysis_artifact(
                    workflow_id=
                        workflow_id,

                    analysis_id=
                        analysis_id,
                )
            )


            assert_equal(
                executed_artifact.source_type,
                "follow_up_prompt",
                "Stored executed artifact changed source type.",
            )

            assert_true(
                isinstance(
                    executed_artifact
                    .pipeline_payload
                    .get(
                        "requested_plan"
                    ),
                    dict,
                ),
                "Executed follow-up lost requested_plan.",
            )

            assert_true(
                isinstance(
                    executed_artifact
                    .pipeline_payload
                    .get(
                        "requested_finding"
                    ),
                    dict,
                ),
                "Executed follow-up lost requested_finding.",
            )

            created_at_before = (
                executed_artifact.created_at_utc
            )


            print(
                "[PASS] Resolution promotes same follow-up artifact"
            )


            reconfigured = (
                reconfigure_requested_analysis_http(
                    RequestedAnalysisReconfigurationRequest(
                        workflow_id=
                            workflow_id,

                        request_id=
                            request_id,

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


            assert_equal(
                reconfigured.analysis_id,
                analysis_id,
                "Reconfiguration changed artifact identity.",
            )

            assert_equal(
                reconfigured.source_type,
                "follow_up_prompt",
                "Reconfiguration changed source type.",
            )


            after_reconfigure = (
                get_analysis_artifact(
                    workflow_id=
                        workflow_id,

                    analysis_id=
                        analysis_id,
                )
            )


            assert_equal(
                after_reconfigure.created_at_utc,
                created_at_before,
                "Reconfiguration changed artifact creation time.",
            )


            stored_plan = (
                after_reconfigure
                .pipeline_payload[
                    "requested_plan"
                ]
            )

            stored_finding = (
                after_reconfigure
                .pipeline_payload[
                    "requested_finding"
                ]
            )


            assert_equal(
                stored_plan[
                    "resolution"
                ][
                    "time_granularity"
                ],
                "week",
                "Reconfigured plan did not persist new granularity.",
            )

            assert_equal(
                stored_plan[
                    "resolution"
                ][
                    "moving_average_window"
                ],
                4,
                "Reconfigured plan did not persist new window.",
            )

            assert_equal(
                stored_finding[
                    "metrics"
                ][
                    "aggregation_period"
                ],
                "week",
                "Reconfigured finding did not persist new period.",
            )

            assert_equal(
                stored_finding[
                    "metrics"
                ][
                    "moving_average_window"
                ],
                4,
                "Reconfigured finding did not persist new window.",
            )


            print(
                "[PASS] Reconfiguration preserves follow-up lifecycle identity"
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


    print()
    print(
        "PASS - Follow-up Requested Analysis bridge v0.1"
    )


if __name__ == "__main__":
    main()

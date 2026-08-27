from __future__ import annotations

import app.reporting.unified_report_artifacts as artifacts


def main() -> None:
    finding = {
        "analysis_id":
            "requested:test-time-series",

        "request_id":
            "request:test-time-series",

        "request_text":
            "CA avec moyenne mobile",

        "title":
            "CA avec moyenne mobile",

        "origin":
            "requested",

        "kind":
            "revenue_moving_average",

        "family":
            "time_series",

        "execution_status":
            "complete",

        "summary":
            [
                "Analyse ex?cut?e."
            ],

        "metrics":
            {
                "aggregation_period":
                    "week",

                "moving_average_window":
                    4,
            },

        "chart_data":
            [
                {
                    "period":
                        "2026-01-05",

                    "value":
                        100.0,

                    "moving_average":
                        100.0,
                }
            ],
    }


    requested_plan = {
        "request_id":
            "request:test-time-series",

        "request_text":
            "CA avec moyenne mobile",

        "kind":
            "revenue_moving_average",

        "status":
            "ready",

        "resolution":
            {
                "resolution_type":
                    "time_series_parameters",

                "time_granularity":
                    "week",

                "moving_average_window":
                    4,
            },

        "source_filename":
            "brief.pdf",

        "source_locator":
            "page 1",

        "source_chunk_id":
            "chunk:1",

        "evidence_unit_id":
            1,
    }


    captured = {}


    original_register = (
        artifacts
        .register_server_owned_analysis
    )


    def fake_register(
        **kwargs
    ):
        captured.update(
            kwargs
        )

        return kwargs


    artifacts.register_server_owned_analysis = (
        fake_register
    )


    try:
        artifacts.register_requested_report_finding(
            workflow_id=
                "workflow:test",

            finding=
                finding,

            requested_plan=
                requested_plan,

            select_by_default=
                True,
        )

    finally:
        artifacts.register_server_owned_analysis = (
            original_register
        )


    payload = (
        captured[
            "pipeline_payload"
        ]
    )


    assert (
        payload[
            "requested_plan"
        ]
        ==
        requested_plan
    )


    assert (
        payload[
            "requested_plan"
        ][
            "resolution"
        ][
            "time_granularity"
        ]
        ==
        "week"
    )


    assert (
        payload[
            "requested_plan"
        ][
            "resolution"
        ][
            "moving_average_window"
        ]
        ==
        4
    )


    assert (
        payload[
            "requested_finding"
        ][
            "metrics"
        ][
            "aggregation_period"
        ]
        ==
        "week"
    )


    assert (
        payload[
            "requested_finding"
        ][
            "metrics"
        ][
            "moving_average_window"
        ]
        ==
        4
    )


    assert (
        captured[
            "source_type"
        ]
        ==
        "document_request"
    )


    assert (
        captured[
            "executed"
        ]
        is True
    )


    print(
        "[PASS] executed requested artifact keeps resolved plan"
    )

    print(
        "[PASS] plan resolution and finding metrics agree on week / 4"
    )

    print(
        "[PASS] requested finding remains server-owned"
    )

    print(
        "[PASS] artifact remains an executed document request"
    )

    print()
    print(
        "PASS - executed requested artifact plan v0.1"
    )


if __name__ == "__main__":
    main()

from __future__ import annotations


from app.reporting.unified_report_artifacts import (
    _synthetic_native_payload,
)


def requested_finding() -> dict:
    return {
        "request_id":
            "request:revenue-moving-average",

        "analysis_id":
            "requested:revenue-moving-average",

        "request_text":
            (
                "Afficher le chiffre d'affaires "
                "avec une moyenne mobile."
            ),

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

        "datasets":
            [
                "dataset:transactions",
            ],

        "analytical_grain":
            "event",

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
                "Le chiffre d'affaires a ?t? agr?g? par semaine.",
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
                        "2026-01-05",

                    "value":
                        1000.0,

                    "moving_average":
                        1000.0,
                },
                {
                    "period":
                        "2026-01-12",

                    "value":
                        1200.0,

                    "moving_average":
                        1100.0,
                },
            ],

        "metrics":
            {
                "aggregation_period":
                    "week",

                "moving_average_window":
                    4,

                "period_count":
                    2,

                "valid_observations":
                    100,
            },

        "source_filename":
            "brief.pdf",

        "source_locator":
            "page 1",

        "page_number":
            1,

        "source_chunk_id":
            "chunk:1",

        "evidence_unit_id":
            1,

        "evidence_quote":
            (
                "chiffre d'affaires avec la moyenne mobile"
            ),

        "adapter_rule_version":
            "requested_report_adapter_v0.2",
    }


def main() -> None:
    finding = (
        requested_finding()
    )


    payload = (
        _synthetic_native_payload(
            artifact_id=
                "analysis:report:test",

            source_type=
                "document_request",

            objective=
                finding[
                    "request_text"
                ],

            finding=
                finding,
        )
    )


    assert (
        payload[
            "analysis_source_type"
        ]
        ==
        "document_request"
    )


    assert (
        payload[
            "requested_finding"
        ]
        ==
        finding
    )


    assert (
        payload[
            "requested_finding"
        ][
            "kind"
        ]
        ==
        "revenue_moving_average"
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


    native_result = (
        payload[
            "items"
        ][
            0
        ][
            "native_tool"
        ][
            "execution"
        ][
            "result"
        ]
    )


    assert (
        native_result[
            "chart_data"
        ]
        ==
        finding[
            "chart_data"
        ]
    )


    automatic_payload = (
        _synthetic_native_payload(
            artifact_id=
                "analysis:report:auto",

            source_type=
                "automatic",

            objective=
                "Analyse automatique",

            finding=
                finding,
        )
    )


    assert (
        automatic_payload[
            "requested_finding"
        ]
        is None
    )


    print(
        "[PASS] document request keeps exact requested finding"
    )

    print(
        "[PASS] requested kind and dynamic time-series metrics preserved"
    )

    print(
        "[PASS] chart payload remains server-owned"
    )

    print(
        "[PASS] automatic analysis does not masquerade as requested finding"
    )

    print()
    print(
        "PASS - document request artifact renderer payload v0.1"
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import pandas as pd

from app.discovery.engine import (
    automatic_time_series_columns,
    build_dataset_profile,
    discover_time_series,
    is_automatic_time_series_axis,
)


def build_test_dataframe(
) -> pd.DataFrame:
    row_count = 24


    return pd.DataFrame(
        {
            "date":
                pd.date_range(
                    "2024-01-01",
                    periods=
                        row_count,
                    freq="MS",
                ),

            # Birth remains a legitimate temporal variable,
            # but represents cohort / individual time rather
            # than the chronology of the observed event.
            "birth":
                [
                    1950
                    +
                    (
                        index
                        %
                        20
                    )
                    *
                    3

                    for index
                    in range(
                        row_count
                    )
                ],

            # At least MIN_NUMERIC_OBSERVATIONS observations
            # are deliberately provided so the production
            # profile exposes price as a quantitative column.
            "price":
                [
                    10.0
                    +
                    index
                    *
                    0.5

                    for index
                    in range(
                        row_count
                    )
                ],
        }
    )



def test_birth_remains_temporal_but_not_longitudinal(
) -> None:
    dataframe = (
        build_test_dataframe()
    )


    profile = (
        build_dataset_profile(
            dataset_id=
                "dataset:test",

            filename=
                "test.csv",

            dataframe=
                dataframe,
        )
    )


    birth_profile = (
        profile.columns[
            "birth"
        ]
    )


    print(
        "birth kind:",
        birth_profile.kind,
    )

    print(
        "birth subtype:",
        birth_profile.analytical_subtype,
    )

    print(
        "birth semantic role:",
        birth_profile.semantic_role,
    )

    print(
        "birth concepts:",
        sorted(
            birth_profile.concepts
        ),
    )


    assert (
        birth_profile.kind
        ==
        "temporal"
    )


    assert (
        birth_profile.analytical_subtype
        ==
        "birth_year"
    )


    assert (
        "birth"
        in profile.temporal_columns
    )


    assert not (
        is_automatic_time_series_axis(
            birth_profile
        )
    )


    print(
        "[PASS] Birth remains temporal but is not "
        "an automatic longitudinal axis"
    )


def test_event_date_remains_time_series_eligible(
) -> None:
    dataframe = (
        build_test_dataframe()
    )


    profile = (
        build_dataset_profile(
            dataset_id=
                "dataset:test",

            filename=
                "test.csv",

            dataframe=
                dataframe,
        )
    )


    date_profile = (
        profile.columns[
            "date"
        ]
    )


    assert (
        date_profile.kind
        ==
        "temporal"
    )


    assert (
        is_automatic_time_series_axis(
            date_profile
        )
    )


    eligible = (
        automatic_time_series_columns(
            profile
        )
    )


    print(
        "All temporal columns:",
        profile.temporal_columns,
    )

    print(
        "Automatic time-series axes:",
        eligible,
    )


    assert (
        "date"
        in eligible
    )


    assert (
        "birth"
        not in eligible
    )


    print(
        "[PASS] Event date remains eligible"
    )


def test_birth_price_time_series_is_not_discovered(
) -> None:
    dataframe = (
        build_test_dataframe()
    )


    profile = (
        build_dataset_profile(
            dataset_id=
                "dataset:test",

            filename=
                "test.csv",

            dataframe=
                dataframe,
        )
    )


    candidates = (
        discover_time_series(
            profile,

            objective=
                None,
        )
    )


    analysis_ids = {
        candidate.analysis_id

        for candidate
        in candidates
    }


    print(
        "Discovered time-series IDs:"
    )


    for analysis_id in sorted(
        analysis_ids
    ):
        print(
            "  -",
            analysis_id,
        )


    assert (
        "dataset:test:time:date:price"
        in analysis_ids
    )


    assert (
        "dataset:test:time:birth:price"
        not in analysis_ids
    )


    print(
        "[PASS] birth x price is excluded from "
        "automatic time-series Discovery"
    )


def test_birth_date_datetime_is_also_excluded(
) -> None:
    dataframe = pd.DataFrame(
        {
            "birth_date":
                pd.to_datetime(
                    [
                        "1980-01-01",
                        "1985-02-02",
                        "1990-03-03",
                        "1995-04-04",
                    ]
                ),

            "value":
                [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                ],
        }
    )


    profile = (
        build_dataset_profile(
            dataset_id=
                "dataset:birth-date",

            filename=
                "birth_date.csv",

            dataframe=
                dataframe,
        )
    )


    birth_date_profile = (
        profile.columns[
            "birth_date"
        ]
    )


    print(
        "birth_date subtype:",
        birth_date_profile.analytical_subtype,
    )

    print(
        "birth_date concepts:",
        sorted(
            birth_date_profile.concepts
        ),
    )


    assert (
        birth_date_profile.kind
        ==
        "temporal"
    )


    assert (
        "age"
        in birth_date_profile.concepts
    )


    assert not (
        is_automatic_time_series_axis(
            birth_date_profile
        )
    )


    print(
        "[PASS] Datetime birth date is also treated "
        "as cohort time rather than longitudinal time"
    )


def main(
) -> None:
    print(
        "=== DATALENS AUTOMATIC TIME-SERIES AXIS v0.1 ==="
    )

    print()


    test_birth_remains_temporal_but_not_longitudinal()

    print()

    test_event_date_remains_time_series_eligible()

    print()

    test_birth_price_time_series_is_not_discovered()

    print()

    test_birth_date_datetime_is_also_excluded()


    print()
    print(
        "PASS - automatic time-series axis v0.1"
    )


if __name__ == "__main__":
    main()

from __future__ import annotations


from pydantic import (
    ValidationError,
)


from app.ml.monitoring_profile import (
    ML_MONITORING_PROFILE_RULE_VERSION,
    MLCategoricalMonitoringBucket,
    MLCategoricalMonitoringFeatureProfile,
    MLMonitoringProfileRecord,
    MLNumericMonitoringFeatureProfile,
)


# ============================================================
# HELPERS
# ============================================================


def expect_validation_error(
    factory,
) -> None:

    try:
        factory()

    except ValidationError:
        return


    raise AssertionError(
        "Expected ValidationError."
    )


def sha(
    character: str,
) -> str:

    return (
        character
        *
        64
    )


def numeric_profile(
    *,
    feature_name: str = "age",
) -> MLNumericMonitoringFeatureProfile:

    return (
        MLNumericMonitoringFeatureProfile(
            feature_name=
                feature_name,

            total_count=
                10,

            non_missing_count=
                8,

            missing_count=
                2,

            missing_rate=
                0.2,

            mean=
                40.0,

            std=
                10.0,

            minimum=
                20.0,

            q25=
                32.0,

            median=
                40.0,

            q75=
                48.0,

            maximum=
                60.0,

            histogram_edges=[
                30.0,
                45.0,
            ],

            histogram_counts=[
                2,
                4,
                2,
            ],

            histogram_rates=[
                0.25,
                0.50,
                0.25,
            ],
        )
    )


def categorical_profile(
    *,
    feature_name: str = "segment",
) -> MLCategoricalMonitoringFeatureProfile:

    return (
        MLCategoricalMonitoringFeatureProfile(
            feature_name=
                feature_name,

            total_count=
                10,

            non_missing_count=
                8,

            missing_count=
                2,

            missing_rate=
                0.2,

            distinct_count=
                3,

            tracked_categories=[
                MLCategoricalMonitoringBucket(
                    value_sha256=
                        sha(
                            "a"
                        ),

                    count=
                        4,

                    rate=
                        0.5,
                ),

                MLCategoricalMonitoringBucket(
                    value_sha256=
                        sha(
                            "b"
                        ),

                    count=
                        2,

                    rate=
                        0.25,
                ),
            ],

            other_count=
                2,

            other_rate=
                0.25,
        )
    )


def monitoring_profile(
) -> MLMonitoringProfileRecord:

    return (
        MLMonitoringProfileRecord(
            profile_id=(
                "monitoring-profile:"
                +
                (
                    "1"
                    *
                    32
                )
            ),

            model_id=(
                "model:"
                +
                (
                    "2"
                    *
                    32
                )
            ),

            workflow_id=
                "prep:monitoring",

            dataset_id=
                "dataset:validated",

            experiment_id=(
                "experiment:"
                +
                (
                    "3"
                    *
                    32
                )
            ),

            preparation_session_revision=
                7,

            training_contract_sha256=
                sha(
                    "c"
                ),

            created_at_utc=
                "2026-08-29T12:00:00+00:00",

            reference_row_count=
                10,

            feature_profiles=[
                numeric_profile(),
                categorical_profile(),
            ],
        )
    )


# ============================================================
# TESTS
# ============================================================


def test_valid_numeric_profile(
) -> None:

    profile = (
        numeric_profile()
    )


    assert (
        profile.kind
        ==
        "numeric"
    )


    assert (
        profile.total_count
        ==
        10
    )


    assert (
        sum(
            profile.histogram_counts
        )
        ==
        8
    )


def test_numeric_counts_must_match(
) -> None:

    expect_validation_error(
        lambda:
            MLNumericMonitoringFeatureProfile(
                feature_name=
                    "age",

                total_count=
                    10,

                non_missing_count=
                    9,

                missing_count=
                    2,

                missing_rate=
                    0.2,

                mean=
                    40.0,

                std=
                    10.0,

                minimum=
                    20.0,

                q25=
                    30.0,

                median=
                    40.0,

                q75=
                    50.0,

                maximum=
                    60.0,

                histogram_counts=[
                    9
                ],

                histogram_rates=[
                    1.0
                ],
            )
    )


def test_numeric_quantiles_must_be_ordered(
) -> None:

    expect_validation_error(
        lambda:
            MLNumericMonitoringFeatureProfile(
                feature_name=
                    "age",

                total_count=
                    10,

                non_missing_count=
                    10,

                missing_count=
                    0,

                missing_rate=
                    0.0,

                mean=
                    40.0,

                std=
                    10.0,

                minimum=
                    20.0,

                q25=
                    50.0,

                median=
                    40.0,

                q75=
                    45.0,

                maximum=
                    60.0,

                histogram_counts=[
                    10
                ],

                histogram_rates=[
                    1.0
                ],
            )
    )


def test_numeric_histogram_shape_fails_closed(
) -> None:

    expect_validation_error(
        lambda:
            MLNumericMonitoringFeatureProfile(
                feature_name=
                    "age",

                total_count=
                    10,

                non_missing_count=
                    10,

                missing_count=
                    0,

                missing_rate=
                    0.0,

                mean=
                    40.0,

                std=
                    10.0,

                minimum=
                    20.0,

                q25=
                    30.0,

                median=
                    40.0,

                q75=
                    50.0,

                maximum=
                    60.0,

                histogram_edges=[
                    30.0,
                    45.0,
                ],

                histogram_counts=[
                    5,
                    5,
                ],

                histogram_rates=[
                    0.5,
                    0.5,
                ],
            )
    )


def test_numeric_histogram_rates_match_counts(
) -> None:

    expect_validation_error(
        lambda:
            MLNumericMonitoringFeatureProfile(
                feature_name=
                    "age",

                total_count=
                    10,

                non_missing_count=
                    10,

                missing_count=
                    0,

                missing_rate=
                    0.0,

                mean=
                    40.0,

                std=
                    10.0,

                minimum=
                    20.0,

                q25=
                    30.0,

                median=
                    40.0,

                q75=
                    50.0,

                maximum=
                    60.0,

                histogram_counts=[
                    4,
                    6,
                ],

                histogram_edges=[
                    40.0
                ],

                histogram_rates=[
                    0.5,
                    0.5,
                ],
            )
    )


def test_valid_hashed_categorical_profile(
) -> None:

    profile = (
        categorical_profile()
    )


    assert (
        profile.kind
        ==
        "categorical"
    )


    assert (
        profile.category_identity
        ==
        "sha256"
    )


    assert (
        len(
            profile.tracked_categories[
                0
            ].value_sha256
        )
        ==
        64
    )


def test_raw_category_value_is_forbidden(
) -> None:

    expect_validation_error(
        lambda:
            MLCategoricalMonitoringBucket(
                value_sha256=
                    sha(
                        "d"
                    ),

                count=
                    1,

                rate=
                    1.0,

                raw_value=
                    "Enterprise",
            )
    )


def test_duplicate_category_hash_fails_closed(
) -> None:

    duplicate = sha(
        "e"
    )


    expect_validation_error(
        lambda:
            MLCategoricalMonitoringFeatureProfile(
                feature_name=
                    "segment",

                total_count=
                    4,

                non_missing_count=
                    4,

                missing_count=
                    0,

                missing_rate=
                    0.0,

                distinct_count=
                    2,

                tracked_categories=[
                    MLCategoricalMonitoringBucket(
                        value_sha256=
                            duplicate,

                        count=
                            2,

                        rate=
                            0.5,
                    ),

                    MLCategoricalMonitoringBucket(
                        value_sha256=
                            duplicate,

                        count=
                            2,

                        rate=
                            0.5,
                    ),
                ],

                other_count=
                    0,

                other_rate=
                    0.0,
            )
    )


def test_categorical_bucket_counts_must_match(
) -> None:

    expect_validation_error(
        lambda:
            MLCategoricalMonitoringFeatureProfile(
                feature_name=
                    "segment",

                total_count=
                    10,

                non_missing_count=
                    10,

                missing_count=
                    0,

                missing_rate=
                    0.0,

                distinct_count=
                    2,

                tracked_categories=[
                    MLCategoricalMonitoringBucket(
                        value_sha256=
                            sha(
                                "f"
                            ),

                        count=
                            4,

                        rate=
                            0.4,
                    ),
                ],

                other_count=
                    5,

                other_rate=
                    0.5,
            )
    )


def test_categorical_bucket_rates_must_match(
) -> None:

    expect_validation_error(
        lambda:
            MLCategoricalMonitoringFeatureProfile(
                feature_name=
                    "segment",

                total_count=
                    10,

                non_missing_count=
                    10,

                missing_count=
                    0,

                missing_rate=
                    0.0,

                distinct_count=
                    2,

                tracked_categories=[
                    MLCategoricalMonitoringBucket(
                        value_sha256=
                            sha(
                                "1"
                            ),

                        count=
                            5,

                        rate=
                            0.4,
                    ),
                ],

                other_count=
                    5,

                other_rate=
                    0.6,
            )
    )


def test_valid_monitoring_profile(
) -> None:

    profile = (
        monitoring_profile()
    )


    assert (
        profile.reference_scope
        ==
        "training_split"
    )


    assert (
        profile.privacy_scope
        ==
        "aggregate_only"
    )


    assert (
        profile.categorical_identity
        ==
        "sha256"
    )


    assert (
        len(
            profile.feature_profiles
        )
        ==
        2
    )


def test_duplicate_feature_names_fail_closed(
) -> None:

    base = (
        monitoring_profile()
        .model_dump(
            mode="json"
        )
    )


    base[
        "feature_profiles"
    ] = [
        numeric_profile(
            feature_name=
                "age"
        ).model_dump(
            mode="json"
        ),

        categorical_profile(
            feature_name=
                "age"
        ).model_dump(
            mode="json"
        ),
    ]


    expect_validation_error(
        lambda:
            MLMonitoringProfileRecord(
                **base
            )
    )


def test_reference_row_count_must_match_features(
) -> None:

    base = (
        monitoring_profile()
        .model_dump(
            mode="json"
        )
    )


    base[
        "reference_row_count"
    ] = 11


    expect_validation_error(
        lambda:
            MLMonitoringProfileRecord(
                **base
            )
    )


def test_raw_authority_payload_is_forbidden(
) -> None:

    base = (
        monitoring_profile()
        .model_dump(
            mode="json"
        )
    )


    base[
        "raw_rows"
    ] = [
        {
            "age":
                42
        }
    ]


    expect_validation_error(
        lambda:
            MLMonitoringProfileRecord(
                **base
            )
    )


def test_monitoring_profile_is_frozen(
) -> None:

    profile = (
        monitoring_profile()
    )


    try:
        profile.reference_row_count = 100

    except ValidationError:
        return


    raise AssertionError(
        "Monitoring profile must be frozen."
    )


def test_rule_version(
) -> None:

    assert (
        ML_MONITORING_PROFILE_RULE_VERSION
        ==
        "ml_monitoring_profile_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS ML MONITORING PROFILE CONTRACT v0.1 ==="
    )


    tests = [
        (
            "Valid numeric monitoring profile",
            test_valid_numeric_profile,
        ),
        (
            "Numeric count consistency",
            test_numeric_counts_must_match,
        ),
        (
            "Numeric quantile ordering",
            test_numeric_quantiles_must_be_ordered,
        ),
        (
            "Numeric histogram shape",
            test_numeric_histogram_shape_fails_closed,
        ),
        (
            "Numeric histogram rate consistency",
            test_numeric_histogram_rates_match_counts,
        ),
        (
            "Valid SHA-256 categorical monitoring profile",
            test_valid_hashed_categorical_profile,
        ),
        (
            "Raw categorical value forbidden",
            test_raw_category_value_is_forbidden,
        ),
        (
            "Duplicate categorical hash blocked",
            test_duplicate_category_hash_fails_closed,
        ),
        (
            "Categorical count consistency",
            test_categorical_bucket_counts_must_match,
        ),
        (
            "Categorical rate consistency",
            test_categorical_bucket_rates_must_match,
        ),
        (
            "Valid monitoring profile record",
            test_valid_monitoring_profile,
        ),
        (
            "Duplicate feature names blocked",
            test_duplicate_feature_names_fail_closed,
        ),
        (
            "Reference row count consistency",
            test_reference_row_count_must_match_features,
        ),
        (
            "Raw monitoring authority payload blocked",
            test_raw_authority_payload_is_forbidden,
        ),
        (
            "Monitoring profile frozen",
            test_monitoring_profile_is_frozen,
        ),
        (
            "Monitoring profile rule version",
            test_rule_version,
        ),
    ]


    for (
        label,
        test,
    ) in tests:

        test()

        print(
            f"[PASS] {label}"
        )


    print()

    print(
        "PASS - ML Monitoring Profile Contract v0.1"
    )


if __name__ == "__main__":
    main()

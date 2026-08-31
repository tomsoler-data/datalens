from __future__ import annotations


import copy


import numpy as np
import pandas as pd


from app.ml.contracts import (
    MLPreprocessingContract,
    MLTrainingContract,
)


from app.ml.experiment_provenance import (
    build_ml_experiment_provenance,
)


from app.ml.model_artifacts import (
    MLModelArtifactRecord,
)


from app.ml.monitoring_profile import (
    MLCategoricalMonitoringFeatureProfile,
    MLNumericMonitoringFeatureProfile,
)


from app.ml.monitoring_profile_builder import (
    ML_MONITORING_MAX_TRACKED_CATEGORIES,
    ML_MONITORING_PROFILE_BUILDER_RULE_VERSION,
    MLMonitoringProfileBuilderError,
    build_ml_monitoring_profile,
    ml_monitoring_category_sha256,
)


# ============================================================
# HELPERS
# ============================================================


MODEL_ID = (
    "model:"
    +
    (
        "1"
        *
        32
    )
)


def expect_builder_error(
    factory,
) -> None:

    try:
        factory()

    except MLMonitoringProfileBuilderError:
        return


    raise AssertionError(
        (
            "Expected "
            "MLMonitoringProfileBuilderError."
        )
    )


def training_contract(
    *,
    feature_columns: list[
        str
    ] | None = None,
    categorical_feature_columns: list[
        str
    ] | None = None,
) -> MLTrainingContract:

    return (
        MLTrainingContract(
            workflow_id=
                "prep:monitoring-builder",

            dataset_id=
                "dataset:validated",

            problem_type=
                "classification",

            target_column=
                "target",

            feature_columns=(
                feature_columns
                if feature_columns
                is not None
                else [
                    "age",
                    "income",
                    "segment",
                ]
            ),

            categorical_feature_columns=(
                categorical_feature_columns
                if categorical_feature_columns
                is not None
                else [
                    "segment"
                ]
            ),

            estimator_key=
                "logistic_regression",

            preprocessing=(
                MLPreprocessingContract(
                    numeric_imputation=
                        "median",

                    categorical_imputation=
                        "most_frequent",

                    categorical_encoding=
                        "one_hot",

                    handle_unknown_categories=
                        "ignore",

                    scale_numeric=
                        True,
                )
            ),
        )
    )


def model_artifact(
    *,
    contract: MLTrainingContract,
    train_rows: int,
    with_provenance: bool = True,
) -> MLModelArtifactRecord:

    metrics = {
        "accuracy":
            0.8,

        "balanced_accuracy":
            0.8,

        "f1_macro":
            0.8,
    }


    provenance = (
        build_ml_experiment_provenance(
            training_contract=
                contract,

            preparation_session_revision=
                7,

            model_id=
                MODEL_ID,

            train_rows=
                train_rows,

            test_rows=
                2,

            metrics=
                metrics,
        )

        if with_provenance

        else None
    )


    return (
        MLModelArtifactRecord(
            model_id=
                MODEL_ID,

            workflow_id=
                contract.workflow_id,

            dataset_id=
                contract.dataset_id,

            training_contract=
                contract,

            experiment_provenance=
                provenance,

            metrics=
                metrics,

            train_rows=
                train_rows,

            test_rows=
                2,

            created_at_utc=
                "2026-08-29T12:00:00+00:00",

            serialization_format=
                "joblib",

            model_path=
                "models/reference.joblib",

            model_file_bytes=
                128,

            model_sha256=(
                "a"
                *
                64
            ),
        )
    )


def mixed_training_frame(
) -> pd.DataFrame:

    return (
        pd.DataFrame(
            {
                "age": [
                    20.0,
                    25.0,
                    30.0,
                    np.nan,
                    40.0,
                    45.0,
                    50.0,
                    55.0,
                    60.0,
                    65.0,
                ],

                "income": [
                    1000.0,
                    1200.0,
                    1400.0,
                    1600.0,
                    1800.0,
                    2000.0,
                    2200.0,
                    2400.0,
                    2600.0,
                    2800.0,
                ],

                "segment": [
                    "standard",
                    "premium",
                    "standard",
                    "standard",
                    "premium",
                    None,
                    "standard",
                    "premium",
                    "standard",
                    "premium",
                ],
            }
        )
    )


def profile_from_frame(
    frame: pd.DataFrame,
    *,
    contract: MLTrainingContract | None = None,
):

    effective_contract = (
        contract
        if contract
        is not None
        else training_contract()
    )


    artifact = (
        model_artifact(
            contract=
                effective_contract,

            train_rows=
                len(
                    frame
                ),
        )
    )


    return (
        build_ml_monitoring_profile(
            x_train=
                frame,

            model_artifact=
                artifact,
        )
    )


# ============================================================
# TESTS
# ============================================================


def test_builder_preserves_training_feature_order(
) -> None:

    frame = (
        mixed_training_frame()
    )


    profile = (
        profile_from_frame(
            frame
        )
    )


    assert (
        [
            feature.feature_name
            for feature
            in profile.feature_profiles
        ]
        ==
        [
            "age",
            "income",
            "segment",
        ]
    )


    assert isinstance(
        profile.feature_profiles[
            0
        ],
        MLNumericMonitoringFeatureProfile,
    )


    assert isinstance(
        profile.feature_profiles[
            2
        ],
        MLCategoricalMonitoringFeatureProfile,
    )


def test_numeric_reference_uses_training_rows_only(
) -> None:

    frame = (
        mixed_training_frame()
    )


    profile = (
        profile_from_frame(
            frame
        )
    )


    age = (
        profile.feature_profiles[
            0
        ]
    )


    assert isinstance(
        age,
        MLNumericMonitoringFeatureProfile,
    )


    assert (
        age.total_count
        ==
        10
    )


    assert (
        age.non_missing_count
        ==
        9
    )


    assert (
        age.missing_count
        ==
        1
    )


    assert (
        age.missing_rate
        ==
        0.1
    )


    assert (
        sum(
            age.histogram_counts
        )
        ==
        9
    )


def test_numeric_histogram_is_deterministic(
) -> None:

    frame = (
        mixed_training_frame()
    )


    first = (
        profile_from_frame(
            frame
        )
    )


    second = (
        profile_from_frame(
            frame
        )
    )


    first_age = (
        first.feature_profiles[
            0
        ]
    )


    second_age = (
        second.feature_profiles[
            0
        ]
    )


    assert (
        first_age
        ==
        second_age
    )


def test_constant_numeric_feature_uses_one_bucket(
) -> None:

    contract = (
        training_contract(
            feature_columns=[
                "constant"
            ],

            categorical_feature_columns=[],
        )
    )


    frame = (
        pd.DataFrame(
            {
                "constant": [
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                ]
            }
        )
    )


    profile = (
        profile_from_frame(
            frame,
            contract=
                contract,
        )
    )


    feature = (
        profile.feature_profiles[
            0
        ]
    )


    assert isinstance(
        feature,
        MLNumericMonitoringFeatureProfile,
    )


    assert (
        feature.histogram_edges
        ==
        []
    )


    assert (
        feature.histogram_counts
        ==
        [
            4
        ]
    )


    assert (
        feature.histogram_rates
        ==
        [
            1.0
        ]
    )


def test_categorical_labels_are_never_persisted(
) -> None:

    frame = (
        mixed_training_frame()
    )


    profile = (
        profile_from_frame(
            frame
        )
    )


    payload = (
        profile.model_dump_json()
    )


    assert (
        "standard"
        not in
        payload
    )


    assert (
        "premium"
        not in
        payload
    )


    categorical = (
        profile.feature_profiles[
            2
        ]
    )


    assert isinstance(
        categorical,
        MLCategoricalMonitoringFeatureProfile,
    )


    assert (
        categorical.distinct_count
        ==
        2
    )


    assert (
        categorical.other_count
        ==
        0
    )


def test_category_identity_is_model_and_feature_bound(
) -> None:

    first = (
        ml_monitoring_category_sha256(
            model_id=(
                "model:"
                +
                (
                    "1"
                    *
                    32
                )
            ),

            feature_name=
                "segment",

            value=
                "premium",
        )
    )


    other_model = (
        ml_monitoring_category_sha256(
            model_id=(
                "model:"
                +
                (
                    "2"
                    *
                    32
                )
            ),

            feature_name=
                "segment",

            value=
                "premium",
        )
    )


    other_feature = (
        ml_monitoring_category_sha256(
            model_id=(
                "model:"
                +
                (
                    "1"
                    *
                    32
                )
            ),

            feature_name=
                "country",

            value=
                "premium",
        )
    )


    assert (
        first
        !=
        other_model
    )


    assert (
        first
        !=
        other_feature
    )


    assert (
        len(
            first
        )
        ==
        64
    )


def test_high_cardinality_categories_are_bounded(
) -> None:

    values = [
        f"category-{index:02d}"

        for index
        in range(
            25
        )
    ]


    contract = (
        training_contract(
            feature_columns=[
                "segment"
            ],

            categorical_feature_columns=[
                "segment"
            ],
        )
    )


    frame = (
        pd.DataFrame(
            {
                "segment":
                    values
            }
        )
    )


    profile = (
        profile_from_frame(
            frame,
            contract=
                contract,
        )
    )


    feature = (
        profile.feature_profiles[
            0
        ]
    )


    assert isinstance(
        feature,
        MLCategoricalMonitoringFeatureProfile,
    )


    assert (
        len(
            feature.tracked_categories
        )
        ==
        ML_MONITORING_MAX_TRACKED_CATEGORIES
    )


    assert (
        feature.distinct_count
        ==
        25
    )


    assert (
        feature.other_count
        ==
        5
    )


    assert (
        feature.other_rate
        ==
        0.2
    )


def test_wrong_training_feature_order_fails_closed(
) -> None:

    frame = (
        mixed_training_frame()[
            [
                "income",
                "age",
                "segment",
            ]
        ]
    )


    contract = (
        training_contract()
    )


    artifact = (
        model_artifact(
            contract=
                contract,

            train_rows=
                len(
                    frame
                ),
        )
    )


    expect_builder_error(
        lambda:
            build_ml_monitoring_profile(
                x_train=
                    frame,

                model_artifact=
                    artifact,
            )
    )


def test_training_row_count_must_match_artifact(
) -> None:

    frame = (
        mixed_training_frame()
    )


    contract = (
        training_contract()
    )


    artifact = (
        model_artifact(
            contract=
                contract,

            train_rows=
                9,
        )
    )


    expect_builder_error(
        lambda:
            build_ml_monitoring_profile(
                x_train=
                    frame,

                model_artifact=
                    artifact,
            )
    )


def test_experiment_provenance_is_required(
) -> None:

    frame = (
        mixed_training_frame()
    )


    contract = (
        training_contract()
    )


    artifact = (
        model_artifact(
            contract=
                contract,

            train_rows=
                len(
                    frame
                ),

            with_provenance=
                False,
        )
    )


    expect_builder_error(
        lambda:
            build_ml_monitoring_profile(
                x_train=
                    frame,

                model_artifact=
                    artifact,
            )
    )


def test_non_dataframe_input_fails_closed(
) -> None:

    contract = (
        training_contract()
    )


    artifact = (
        model_artifact(
            contract=
                contract,

            train_rows=
                1,
        )
    )


    expect_builder_error(
        lambda:
            build_ml_monitoring_profile(
                x_train=[
                    {
                        "age":
                            20
                    }
                ],

                model_artifact=
                    artifact,
            )
    )


def test_non_finite_numeric_value_fails_closed(
) -> None:

    frame = (
        mixed_training_frame()
    )


    frame.loc[
        0,
        "income"
    ] = np.inf


    contract = (
        training_contract()
    )


    artifact = (
        model_artifact(
            contract=
                contract,

            train_rows=
                len(
                    frame
                ),
        )
    )


    expect_builder_error(
        lambda:
            build_ml_monitoring_profile(
                x_train=
                    frame,

                model_artifact=
                    artifact,
            )
    )


def test_mixed_categorical_python_families_fail_closed(
) -> None:

    contract = (
        training_contract(
            feature_columns=[
                "segment"
            ],

            categorical_feature_columns=[
                "segment"
            ],
        )
    )


    frame = (
        pd.DataFrame(
            {
                "segment": [
                    "premium",
                    1,
                    "standard",
                    2,
                ]
            },
            dtype=object,
        )
    )


    artifact = (
        model_artifact(
            contract=
                contract,

            train_rows=
                len(
                    frame
                ),
        )
    )


    expect_builder_error(
        lambda:
            build_ml_monitoring_profile(
                x_train=
                    frame,

                model_artifact=
                    artifact,
            )
    )


def test_builder_does_not_mutate_training_frame(
) -> None:

    frame = (
        mixed_training_frame()
    )


    before = (
        frame.copy(
            deep=True
        )
    )


    _ = (
        profile_from_frame(
            frame
        )
    )


    pd.testing.assert_frame_equal(
        frame,
        before,
    )


def test_profile_identity_is_bound_to_model_provenance(
) -> None:

    frame = (
        mixed_training_frame()
    )


    contract = (
        training_contract()
    )


    artifact = (
        model_artifact(
            contract=
                contract,

            train_rows=
                len(
                    frame
                ),
        )
    )


    profile = (
        build_ml_monitoring_profile(
            x_train=
                frame,

            model_artifact=
                artifact,
        )
    )


    provenance = (
        artifact
        .experiment_provenance
    )


    assert (
        provenance
        is not None
    )


    assert (
        profile.model_id
        ==
        artifact.model_id
    )


    assert (
        profile.workflow_id
        ==
        artifact.workflow_id
    )


    assert (
        profile.dataset_id
        ==
        artifact.dataset_id
    )


    assert (
        profile.experiment_id
        ==
        provenance.experiment_id
    )


    assert (
        profile.preparation_session_revision
        ==
        provenance.preparation_session_revision
    )


    assert (
        profile.training_contract_sha256
        ==
        provenance.training_contract_sha256
    )


    assert (
        profile.reference_scope
        ==
        "training_split"
    )


    assert (
        profile.reference_row_count
        ==
        artifact.train_rows
    )


def test_builder_rule_version(
) -> None:

    assert (
        ML_MONITORING_PROFILE_BUILDER_RULE_VERSION
        ==
        "ml_monitoring_profile_builder_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS ML MONITORING PROFILE BUILDER v0.1 ==="
    )


    tests = [
        (
            "Training feature order preserved",
            test_builder_preserves_training_feature_order,
        ),
        (
            "Numeric reference uses training rows",
            test_numeric_reference_uses_training_rows_only,
        ),
        (
            "Numeric histogram deterministic",
            test_numeric_histogram_is_deterministic,
        ),
        (
            "Constant numeric feature",
            test_constant_numeric_feature_uses_one_bucket,
        ),
        (
            "Categorical raw labels absent",
            test_categorical_labels_are_never_persisted,
        ),
        (
            "Model-bound categorical identity",
            test_category_identity_is_model_and_feature_bound,
        ),
        (
            "High-cardinality categorical bound",
            test_high_cardinality_categories_are_bounded,
        ),
        (
            "Wrong feature order blocked",
            test_wrong_training_feature_order_fails_closed,
        ),
        (
            "Training row count binding",
            test_training_row_count_must_match_artifact,
        ),
        (
            "Experiment provenance required",
            test_experiment_provenance_is_required,
        ),
        (
            "Non-DataFrame input blocked",
            test_non_dataframe_input_fails_closed,
        ),
        (
            "Non-finite numeric blocked",
            test_non_finite_numeric_value_fails_closed,
        ),
        (
            "Mixed categorical families blocked",
            test_mixed_categorical_python_families_fail_closed,
        ),
        (
            "Training frame remains immutable",
            test_builder_does_not_mutate_training_frame,
        ),
        (
            "Profile bound to model provenance",
            test_profile_identity_is_bound_to_model_provenance,
        ),
        (
            "Builder rule version",
            test_builder_rule_version,
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
        "PASS - ML Monitoring Profile Builder v0.1"
    )


if __name__ == "__main__":
    main()

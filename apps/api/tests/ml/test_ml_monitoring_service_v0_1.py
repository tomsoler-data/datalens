from __future__ import annotations


import inspect


from contextlib import (
    contextmanager,
)


from dataclasses import (
    dataclass,
)


import pandas as pd


import app.ml.monitoring_service as service_module


from app.ml.drift_evaluation_store import (
    list_ml_drift_evaluations_for_model,
)


from app.ml.monitoring_profile_store import (
    register_ml_monitoring_profile,
)


from app.ml.monitoring_service import (
    ML_MONITORING_SERVICE_RULE_VERSION,
    MLMonitoringObservedDatasetError,
    MLMonitoringServiceAuthorityError,
    MLMonitoringServiceExecutionError,
    run_ml_monitoring,
)


from app.persistence.sqlite_database import (
    sqlite_connection,
)


from tests.ml.test_ml_monitoring_profile_store_v0_1 import (
    isolated_environment,
    persisted_artifact_and_profile,
    training_frame,
)


# ============================================================
# FAKE SERVER-OWNED HANDOFF
# ============================================================


@dataclass(
    frozen=True
)
class FakeMonitoringHandoff:

    workflow_id: str

    session_revision: int

    dataset_ids: tuple[
        str,
        ...
    ]

    dataset_records: tuple[
        dict,
        ...
    ]


# ============================================================
# HELPERS
# ============================================================


def expect_error(
    error_type,
    factory,
) -> None:

    try:
        factory()

    except error_type:
        return


    raise AssertionError(
        (
            "Expected "
            f"{error_type.__name__}."
        )
    )


def persisted_authority(
    *,
    register_profile: bool = True,
):

    (
        artifact,
        profile,
    ) = (
        persisted_artifact_and_profile()
    )


    if register_profile:
        register_ml_monitoring_profile(
            profile=
                profile
        )


    return (
        artifact,
        profile,
    )


@contextmanager
def patched_handoff(
    *,
    dataframe: pd.DataFrame,
    workflow_id: str,
    dataset_id: str = "dataset:observed",
    session_revision: int = 7,
    authorized_dataset_ids: (
        tuple[
            str,
            ...
        ]
        |
        None
    ) = None,
    dataset_records: (
        tuple[
            dict,
            ...
        ]
        |
        None
    ) = None,
):

    original = (
        service_module
        .load_validated_analysis_input
    )


    effective_ids = (
        authorized_dataset_ids

        if authorized_dataset_ids
        is not None

        else (
            dataset_id,
        )
    )


    effective_records = (
        dataset_records

        if dataset_records
        is not None

        else (
            {
                "dataset_id":
                    dataset_id,

                "dataframe":
                    dataframe.copy(
                        deep=True
                    ),
            },
        )
    )


    def fake_load_validated_analysis_input(
        *,
        workflow_id: str,
    ):
        return (
            FakeMonitoringHandoff(
                workflow_id=
                    workflow_id,

                session_revision=
                    session_revision,

                dataset_ids=
                    effective_ids,

                dataset_records=
                    effective_records,
            )
        )


    service_module.load_validated_analysis_input = (
        fake_load_validated_analysis_input
    )


    try:
        yield

    finally:
        service_module.load_validated_analysis_input = (
            original
        )


def observed_frame_with_extras(
) -> pd.DataFrame:

    frame = (
        training_frame()
        .copy(
            deep=True
        )
    )


    frame[
        "target"
    ] = [
        0,
        1,
        0,
        1,
        0,
        1,
    ]


    frame[
        "not_a_model_feature"
    ] = [
        "private-extra-value"
        for _
        in range(
            len(
                frame
            )
        )
    ]


    return frame


# ============================================================
# PUBLIC TRUST SURFACE
# ============================================================


def test_service_accepts_only_identifiers(
) -> None:

    signature = inspect.signature(
        run_ml_monitoring
    )


    assert (
        set(
            signature.parameters
        )
        ==
        {
            "workflow_id",
            "model_id",
            "observed_dataset_id",
        }
    )


# ============================================================
# SUCCESS
# ============================================================


def test_server_owned_monitoring_roundtrip(
) -> None:

    with isolated_environment():

        (
            artifact,
            profile,
        ) = persisted_authority()


        observed = (
            observed_frame_with_extras()
        )


        with patched_handoff(
            dataframe=
                observed,

            workflow_id=
                artifact.workflow_id,
        ):

            result = (
                run_ml_monitoring(
                    workflow_id=
                        artifact.workflow_id,

                    model_id=
                        artifact.model_id,

                    observed_dataset_id=
                        "dataset:observed",
                )
            )


        assert (
            result.model_id
            ==
            artifact.model_id
        )


        assert (
            result.profile_id
            ==
            profile.profile_id
        )


        assert (
            result.workflow_id
            ==
            artifact.workflow_id
        )


        assert (
            result.observed_dataset_id
            ==
            "dataset:observed"
        )


        assert (
            result
            .observed_preparation_session_revision
            ==
            7
        )


        assert [
            feature.feature_name
            for feature
            in result.feature_results
        ] == [
            "age",
            "segment",
        ]


        history = (
            list_ml_drift_evaluations_for_model(
                model_id=
                    artifact.model_id,

                workflow_id=
                    artifact.workflow_id,
            )
        )


        assert (
            history
            ==
            [
                result
            ]
        )


# ============================================================
# EXTRA COLUMNS
# ============================================================


def test_target_and_extra_columns_are_not_monitored(
) -> None:

    with isolated_environment():

        (
            artifact,
            _,
        ) = persisted_authority()


        secret = (
            "THIS-MUST-NOT-ENTER-DRIFT-RESULT"
        )


        observed = (
            observed_frame_with_extras()
        )


        observed[
            "not_a_model_feature"
        ] = [
            secret
            for _
            in range(
                len(
                    observed
                )
            )
        ]


        with patched_handoff(
            dataframe=
                observed,

            workflow_id=
                artifact.workflow_id,
        ):

            result = (
                run_ml_monitoring(
                    workflow_id=
                        artifact.workflow_id,

                    model_id=
                        artifact.model_id,

                    observed_dataset_id=
                        "dataset:observed",
                )
            )


        payload = str(
            result.model_dump(
                mode="json"
            )
        )


        assert (
            "target"
            not in
            [
                feature.feature_name
                for feature
                in result.feature_results
            ]
        )


        assert (
            "not_a_model_feature"
            not in
            payload
        )


        assert (
            secret
            not in
            payload
        )


# ============================================================
# OBSERVED DATASET AUTHORITY
# ============================================================


def test_dataset_outside_validated_handoff_is_blocked(
) -> None:

    with isolated_environment():

        (
            artifact,
            _,
        ) = persisted_authority()


        observed = (
            training_frame()
        )


        with patched_handoff(
            dataframe=
                observed,

            workflow_id=
                artifact.workflow_id,

            dataset_id=
                "dataset:authorized",
        ):

            expect_error(
                MLMonitoringObservedDatasetError,

                lambda:
                    run_ml_monitoring(
                        workflow_id=
                            artifact.workflow_id,

                        model_id=
                            artifact.model_id,

                        observed_dataset_id=
                            "dataset:not-authorized",
                    ),
            )


        assert (
            list_ml_drift_evaluations_for_model(
                model_id=
                    artifact.model_id,

                workflow_id=
                    artifact.workflow_id,
            )
            ==
            []
        )


# ============================================================
# MODEL / WORKFLOW AUTHORITY
# ============================================================


def test_cross_workflow_model_request_is_blocked(
) -> None:

    with isolated_environment():

        (
            artifact,
            _,
        ) = persisted_authority()


        expect_error(
            MLMonitoringServiceAuthorityError,

            lambda:
                run_ml_monitoring(
                    workflow_id=
                        "prep:other-workflow",

                    model_id=
                        artifact.model_id,

                    observed_dataset_id=
                        "dataset:observed",
                ),
        )


# ============================================================
# MONITORING PROFILE AUTHORITY
# ============================================================


def test_missing_monitoring_profile_is_blocked(
) -> None:

    with isolated_environment():

        (
            artifact,
            _,
        ) = persisted_authority(
            register_profile=
                False
        )


        expect_error(
            MLMonitoringServiceAuthorityError,

            lambda:
                run_ml_monitoring(
                    workflow_id=
                        artifact.workflow_id,

                    model_id=
                        artifact.model_id,

                    observed_dataset_id=
                        "dataset:observed",
                ),
        )


# ============================================================
# FEATURE SURFACE
# ============================================================


def test_missing_required_feature_is_blocked(
) -> None:

    with isolated_environment():

        (
            artifact,
            _,
        ) = persisted_authority()


        observed = (
            training_frame()
            .drop(
                columns=[
                    "segment"
                ]
            )
        )


        with patched_handoff(
            dataframe=
                observed,

            workflow_id=
                artifact.workflow_id,
        ):

            expect_error(
                MLMonitoringObservedDatasetError,

                lambda:
                    run_ml_monitoring(
                        workflow_id=
                            artifact.workflow_id,

                        model_id=
                            artifact.model_id,

                        observed_dataset_id=
                            "dataset:observed",
                    ),
            )


        assert (
            list_ml_drift_evaluations_for_model(
                model_id=
                    artifact.model_id,

                workflow_id=
                    artifact.workflow_id,
            )
            ==
            []
        )


def test_duplicate_required_feature_is_blocked(
) -> None:

    with isolated_environment():

        (
            artifact,
            _,
        ) = persisted_authority()


        base = (
            training_frame()
        )


        observed = pd.concat(
            [
                base[
                    [
                        "age"
                    ]
                ],
                base[
                    [
                        "age"
                    ]
                ],
                base[
                    [
                        "segment"
                    ]
                ],
            ],
            axis=1,
        )


        assert list(
            observed.columns
        ) == [
            "age",
            "age",
            "segment",
        ]


        with patched_handoff(
            dataframe=
                observed,

            workflow_id=
                artifact.workflow_id,
        ):

            expect_error(
                MLMonitoringObservedDatasetError,

                lambda:
                    run_ml_monitoring(
                        workflow_id=
                            artifact.workflow_id,

                        model_id=
                            artifact.model_id,

                        observed_dataset_id=
                            "dataset:observed",
                    ),
            )


# ============================================================
# OBSERVED DATA IMMUTABILITY
# ============================================================


def test_observed_handoff_dataframe_remains_immutable(
) -> None:

    with isolated_environment():

        (
            artifact,
            _,
        ) = persisted_authority()


        observed = (
            observed_frame_with_extras()
        )


        before = (
            observed.copy(
                deep=True
            )
        )


        with patched_handoff(
            dataframe=
                observed,

            workflow_id=
                artifact.workflow_id,
        ):

            run_ml_monitoring(
                workflow_id=
                    artifact.workflow_id,

                model_id=
                    artifact.model_id,

                observed_dataset_id=
                    "dataset:observed",
            )


        pd.testing.assert_frame_equal(
            observed,
            before,
        )


# ============================================================
# LATE PREPARATION REVISION RACE
# ============================================================


def test_late_preparation_revision_race_is_blocked(
) -> None:

    with isolated_environment():

        (
            artifact,
            _,
        ) = persisted_authority()


        observed = (
            training_frame()
        )


        original_register = (
            service_module
            .register_ml_drift_evaluation
        )


        def racing_register(
            *,
            evaluation,
        ):

            with sqlite_connection(
                write=True
            ) as connection:

                connection.execute(
                    """
                    UPDATE preparation_sessions

                    SET revision = ?

                    WHERE
                        workflow_id = ?
                    """,
                    (
                        8,
                        artifact.workflow_id,
                    ),
                )


            return (
                original_register(
                    evaluation=
                        evaluation
                )
            )


        service_module.register_ml_drift_evaluation = (
            racing_register
        )


        try:

            with patched_handoff(
                dataframe=
                    observed,

                workflow_id=
                    artifact.workflow_id,

                session_revision=
                    7,
            ):

                expect_error(
                    MLMonitoringServiceExecutionError,

                    lambda:
                        run_ml_monitoring(
                            workflow_id=
                                artifact.workflow_id,

                            model_id=
                                artifact.model_id,

                            observed_dataset_id=
                                "dataset:observed",
                        ),
                )

        finally:
            service_module.register_ml_drift_evaluation = (
                original_register
            )


        with sqlite_connection(
            write=False
        ) as connection:

            count = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM ml_drift_evaluations

                    WHERE
                        model_id = ?
                    """,
                    (
                        artifact.model_id,
                    ),
                )
                .fetchone()[0]
            )


        assert (
            count
            ==
            0
        )


# ============================================================
# MALFORMED HANDOFF
# ============================================================


def test_handoff_scope_record_mismatch_is_blocked(
) -> None:

    with isolated_environment():

        (
            artifact,
            _,
        ) = persisted_authority()


        observed = (
            training_frame()
        )


        with patched_handoff(
            dataframe=
                observed,

            workflow_id=
                artifact.workflow_id,

            authorized_dataset_ids=(
                "dataset:observed",
                "dataset:missing-record",
            ),

            dataset_records=(
                {
                    "dataset_id":
                        "dataset:observed",

                    "dataframe":
                        observed.copy(
                            deep=True
                        ),
                },
            ),
        ):

            expect_error(
                MLMonitoringServiceAuthorityError,

                lambda:
                    run_ml_monitoring(
                        workflow_id=
                            artifact.workflow_id,

                        model_id=
                            artifact.model_id,

                        observed_dataset_id=
                            "dataset:observed",
                    ),
            )


# ============================================================
# VERSION
# ============================================================


def test_rule_version(
) -> None:

    assert (
        ML_MONITORING_SERVICE_RULE_VERSION
        ==
        "ml_monitoring_service_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS ML MONITORING SERVICE v0.1 ==="
    )

    print()


    tests = [
        (
            "Public service accepts identifiers only",
            test_service_accepts_only_identifiers,
        ),
        (
            "Server-owned monitoring roundtrip",
            test_server_owned_monitoring_roundtrip,
        ),
        (
            "Target / extra columns excluded",
            test_target_and_extra_columns_are_not_monitored,
        ),
        (
            "Observed dataset Handoff authority",
            test_dataset_outside_validated_handoff_is_blocked,
        ),
        (
            "Cross-workflow Model Artifact blocked",
            test_cross_workflow_model_request_is_blocked,
        ),
        (
            "Missing Monitoring Profile blocked",
            test_missing_monitoring_profile_is_blocked,
        ),
        (
            "Missing required feature blocked",
            test_missing_required_feature_is_blocked,
        ),
        (
            "Duplicate required feature blocked",
            test_duplicate_required_feature_is_blocked,
        ),
        (
            "Observed DataFrame remains immutable",
            test_observed_handoff_dataframe_remains_immutable,
        ),
        (
            "Late Preparation revision race blocked",
            test_late_preparation_revision_race_is_blocked,
        ),
        (
            "Malformed Handoff scope blocked",
            test_handoff_scope_record_mismatch_is_blocked,
        ),
        (
            "Monitoring Service rule version",
            test_rule_version,
        ),
    ]


    for (
        label,
        callback,
    ) in tests:

        callback()

        print(
            f"[PASS] {label}"
        )


    print()

    print(
        "PASS - ML Monitoring Service v0.1"
    )


if __name__ == "__main__":
    main()

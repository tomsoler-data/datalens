from __future__ import annotations


import math


import pandas as pd


from app.ml.drift_evaluation_store import (
    list_ml_drift_evaluations_for_model,
)


from app.ml.monitoring_service import (
    ML_MONITORING_SERVICE_RULE_VERSION,
    MLMonitoringServiceAuthorityError,
    MLMonitoringServiceExecutionError,
    MLMonitoringServiceInputError,
    run_ml_monitoring,
)


from tests.ml.test_ml_monitoring_profile_store_v0_1 import (
    isolated_environment,
    training_frame,
)


from tests.ml.test_ml_monitoring_service_v0_1 import (
    patched_handoff,
    persisted_authority,
)


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


# ============================================================
# EMPTY IDENTITIES
# ============================================================


def test_empty_workflow_id_blocked(
) -> None:

    expect_error(
        MLMonitoringServiceInputError,

        lambda:
            run_ml_monitoring(
                workflow_id="   ",
                model_id="model:any",
                observed_dataset_id="dataset:any",
            ),
    )


def test_empty_model_id_blocked(
) -> None:

    expect_error(
        MLMonitoringServiceInputError,

        lambda:
            run_ml_monitoring(
                workflow_id="prep:any",
                model_id="   ",
                observed_dataset_id="dataset:any",
            ),
    )


def test_empty_observed_dataset_id_blocked(
) -> None:

    expect_error(
        MLMonitoringServiceInputError,

        lambda:
            run_ml_monitoring(
                workflow_id="prep:any",
                model_id="model:any",
                observed_dataset_id="   ",
            ),
    )


# ============================================================
# INVALID HANDOFF REVISION
# ============================================================


def test_missing_handoff_revision_blocked(
) -> None:

    with isolated_environment():

        (
            artifact,
            _,
        ) = persisted_authority()


        with patched_handoff(
            dataframe=
                training_frame(),

            workflow_id=
                artifact.workflow_id,

            session_revision=
                None,
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


def test_boolean_handoff_revision_blocked(
) -> None:

    with isolated_environment():

        (
            artifact,
            _,
        ) = persisted_authority()


        with patched_handoff(
            dataframe=
                training_frame(),

            workflow_id=
                artifact.workflow_id,

            session_revision=
                True,
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


def test_negative_handoff_revision_blocked(
) -> None:

    with isolated_environment():

        (
            artifact,
            _,
        ) = persisted_authority()


        with patched_handoff(
            dataframe=
                training_frame(),

            workflow_id=
                artifact.workflow_id,

            session_revision=
                -1,
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
# HANDOFF IDENTITY INTEGRITY
# ============================================================


def test_duplicate_handoff_dataset_ids_blocked(
) -> None:

    with isolated_environment():

        (
            artifact,
            _,
        ) = persisted_authority()


        with patched_handoff(
            dataframe=
                training_frame(),

            workflow_id=
                artifact.workflow_id,

            authorized_dataset_ids=(
                "dataset:observed",
                "dataset:observed",
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


def test_duplicate_handoff_records_blocked(
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

            dataset_records=(
                {
                    "dataset_id":
                        "dataset:observed",

                    "dataframe":
                        observed.copy(
                            deep=True
                        ),
                },
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


def test_non_dataframe_handoff_payload_blocked(
) -> None:

    with isolated_environment():

        (
            artifact,
            _,
        ) = persisted_authority()


        with patched_handoff(
            dataframe=
                training_frame(),

            workflow_id=
                artifact.workflow_id,

            dataset_records=(
                {
                    "dataset_id":
                        "dataset:observed",

                    "dataframe":
                        [
                            {
                                "age": 20
                            }
                        ],
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
# EVALUATOR FAILURE MUST NOT PERSIST
# ============================================================


def test_invalid_observed_numeric_data_does_not_persist(
) -> None:

    with isolated_environment():

        (
            artifact,
            _,
        ) = persisted_authority()


        observed = (
            training_frame()
            .copy(
                deep=True
            )
        )


        observed.loc[
            0,
            "age",
        ] = math.inf


        with patched_handoff(
            dataframe=
                observed,

            workflow_id=
                artifact.workflow_id,
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
# ALL-MISSING MUST FAIL CLOSED AS DRIFT, NOT CRASH
# ============================================================


def test_all_missing_numeric_reaches_drift_result(
) -> None:

    with isolated_environment():

        (
            artifact,
            _,
        ) = persisted_authority()


        observed = (
            training_frame()
            .copy(
                deep=True
            )
        )


        observed[
            "age"
        ] = pd.Series(
            [
                None
                for _
                in range(
                    len(
                        observed
                    )
                )
            ],
            dtype="float64",
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


        age_result = next(
            item

            for item
            in result.feature_results

            if (
                item.feature_name
                ==
                "age"
            )
        )


        assert (
            age_result.status
            ==
            "drift"
        )


        assert (
            age_result.distribution_status
            ==
            "not_evaluable"
        )


        assert (
            result.overall_status
            ==
            "drift"
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
        (
            "=== DATALENS ML MONITORING "
            "SERVICE HARDENING v0.1 ==="
        )
    )

    print()


    tests = [
        (
            "Empty workflow identity blocked",
            test_empty_workflow_id_blocked,
        ),
        (
            "Empty model identity blocked",
            test_empty_model_id_blocked,
        ),
        (
            "Empty observed dataset identity blocked",
            test_empty_observed_dataset_id_blocked,
        ),
        (
            "Missing Handoff revision blocked",
            test_missing_handoff_revision_blocked,
        ),
        (
            "Boolean Handoff revision blocked",
            test_boolean_handoff_revision_blocked,
        ),
        (
            "Negative Handoff revision blocked",
            test_negative_handoff_revision_blocked,
        ),
        (
            "Duplicate Handoff dataset identities blocked",
            test_duplicate_handoff_dataset_ids_blocked,
        ),
        (
            "Duplicate Handoff records blocked",
            test_duplicate_handoff_records_blocked,
        ),
        (
            "Non-DataFrame Handoff payload blocked",
            test_non_dataframe_handoff_payload_blocked,
        ),
        (
            "Invalid observed numeric data leaves no history",
            test_invalid_observed_numeric_data_does_not_persist,
        ),
        (
            "All-missing numeric remains fail-closed drift",
            test_all_missing_numeric_reaches_drift_result,
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
        (
            "PASS - ML Monitoring "
            "Service Hardening v0.1"
        )
    )


if __name__ == "__main__":
    main()

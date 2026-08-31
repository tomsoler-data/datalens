from __future__ import annotations

import math

from datetime import (
    datetime,
    timezone,
)

from uuid import (
    uuid4,
)

import numpy as np
import pandas as pd

from app.ml.drift_evaluation import (
    MLCategoricalDriftFeatureResult,
    MLDriftEvaluationRecord,
    MLNumericDriftFeatureResult,
    ML_DRIFT_MISSING_RATE_DRIFT_THRESHOLD,
    ML_DRIFT_MISSING_RATE_WARNING_THRESHOLD,
    ML_DRIFT_OUTSIDE_RANGE_DRIFT_THRESHOLD,
    ML_DRIFT_OUTSIDE_RANGE_WARNING_THRESHOLD,
    ML_DRIFT_PSI_EPSILON,
    ML_DRIFT_EVALUATION_RULE_VERSION,
    ml_drift_max_status,
    ml_drift_status_for_psi,
    ml_drift_status_for_rate_shift,
)

from app.ml.model_artifacts import (
    MLModelArtifactRecord,
)

from app.ml.monitoring_profile import (
    MLCategoricalMonitoringFeatureProfile,
    MLMonitoringProfileRecord,
    MLNumericMonitoringFeatureProfile,
)

from app.ml.monitoring_profile_builder import (
    MLMonitoringProfileBuilderError,
    ml_monitoring_category_sha256,
)


# ============================================================
# VERSION
# ============================================================


ML_DRIFT_EVALUATOR_RULE_VERSION = (
    "ml_drift_evaluator_v0.1"
)


# ============================================================
# ERRORS
# ============================================================


class MLDriftEvaluatorError(
    RuntimeError
):
    pass


# ============================================================
# TIME / ID
# ============================================================


def _utc_now_iso() -> str:
    return (
        datetime.now(
            timezone.utc
        )
        .isoformat()
    )


def _new_evaluation_id() -> str:
    return (
        "drift-evaluation:"
        +
        uuid4().hex
    )


# ============================================================
# PSI
# ============================================================


def _population_stability_index(
    *,
    reference_rates: list[
        float
    ],
    observed_rates: list[
        float
    ],
) -> float:

    if (
        not reference_rates
        or
        len(reference_rates)
        !=
        len(observed_rates)
    ):
        raise MLDriftEvaluatorError(
            (
                "PSI requires non-empty "
                "reference/observed distributions "
                "with identical shape."
            )
        )

    score = 0.0

    for (
        reference_rate,
        observed_rate,
    ) in zip(
        reference_rates,
        observed_rates,
    ):
        reference = max(
            float(reference_rate),
            ML_DRIFT_PSI_EPSILON,
        )

        observed = max(
            float(observed_rate),
            ML_DRIFT_PSI_EPSILON,
        )

        score += (
            (
                observed
                -
                reference
            )
            *
            math.log(
                observed
                /
                reference
            )
        )

    if not math.isfinite(
        score
    ):
        raise MLDriftEvaluatorError(
            "PSI produced a non-finite score."
        )

    return max(
        0.0,
        float(score),
    )


# ============================================================
# AUTHORITY
# ============================================================


def _validate_authority(
    *,
    profile: MLMonitoringProfileRecord,
    artifact: MLModelArtifactRecord,
) -> None:

    provenance = (
        artifact.experiment_provenance
    )

    if provenance is None:
        raise MLDriftEvaluatorError(
            (
                "ML Drift Evaluation requires "
                "trusted Experiment Provenance."
            )
        )

    bindings = [
        (
            "model_id",
            profile.model_id,
            artifact.model_id,
        ),
        (
            "workflow_id",
            profile.workflow_id,
            artifact.workflow_id,
        ),
        (
            "dataset_id",
            profile.dataset_id,
            artifact.dataset_id,
        ),
        (
            "experiment_id",
            profile.experiment_id,
            provenance.experiment_id,
        ),
        (
            "preparation_session_revision",
            profile.preparation_session_revision,
            provenance.preparation_session_revision,
        ),
        (
            "training_contract_sha256",
            profile.training_contract_sha256,
            provenance.training_contract_sha256,
        ),
        (
            "reference_row_count",
            profile.reference_row_count,
            artifact.train_rows,
        ),
    ]

    for (
        name,
        profile_value,
        authority_value,
    ) in bindings:
        if (
            profile_value
            !=
            authority_value
        ):
            raise MLDriftEvaluatorError(
                (
                    "Monitoring Profile does not "
                    "match trusted Model Artifact "
                    "authority. "
                    f"field={name}"
                )
            )

    expected_features = list(
        artifact
        .training_contract
        .feature_columns
    )

    actual_features = [
        feature.feature_name
        for feature
        in profile.feature_profiles
    ]

    if (
        actual_features
        !=
        expected_features
    ):
        raise MLDriftEvaluatorError(
            (
                "Monitoring Profile feature order "
                "does not match the persisted "
                "Training Contract."
            )
        )

    categorical_features = set(
        artifact
        .training_contract
        .categorical_feature_columns
    )

    for feature in profile.feature_profiles:
        expected_kind = (
            "categorical"
            if (
                feature.feature_name
                in
                categorical_features
            )
            else "numeric"
        )

        if feature.kind != expected_kind:
            raise MLDriftEvaluatorError(
                (
                    "Monitoring Profile feature "
                    "kind does not match the "
                    "Training Contract. "
                    f"feature={feature.feature_name}"
                )
            )


# ============================================================
# OBSERVED FRAME VALIDATION
# ============================================================


def _categorical_family(
    value: object,
) -> str:

    if isinstance(
        value,
        (
            bool,
            np.bool_,
        ),
    ):
        return "boolean"

    if isinstance(
        value,
        str,
    ):
        return "text"

    if isinstance(
        value,
        (
            int,
            float,
            np.integer,
            np.floating,
        ),
    ):
        return "numeric"

    return (
        type(value).__name__
    )


def _validate_observed_frame(
    *,
    features: pd.DataFrame,
    artifact: MLModelArtifactRecord,
    profile: MLMonitoringProfileRecord,
) -> pd.DataFrame:

    if not isinstance(
        features,
        pd.DataFrame,
    ):
        raise MLDriftEvaluatorError(
            (
                "ML Drift Evaluation input must "
                "be a pandas DataFrame."
            )
        )

    if features.empty:
        raise MLDriftEvaluatorError(
            (
                "ML Drift Evaluation observed "
                "frame cannot be empty."
            )
        )

    expected_columns = list(
        artifact
        .training_contract
        .feature_columns
    )

    if (
        list(features.columns)
        !=
        expected_columns
    ):
        raise MLDriftEvaluatorError(
            (
                "Observed feature columns do not "
                "match the ordered Training "
                "Contract feature surface."
            )
        )

    categorical_features = set(
        artifact
        .training_contract
        .categorical_feature_columns
    )

    profiles_by_name = {
        feature.feature_name:
            feature
        for feature
        in profile.feature_profiles
    }

    for feature_name in expected_columns:
        series = features[
            feature_name
        ]

        reference = (
            profiles_by_name[
                feature_name
            ]
        )

        missing_mask = (
            series.isna()
        )

        observed = (
            series[
                ~missing_mask
            ]
        )

        if (
            feature_name
            in
            categorical_features
        ):
            if not isinstance(
                reference,
                MLCategoricalMonitoringFeatureProfile,
            ):
                raise MLDriftEvaluatorError(
                    (
                        "Categorical observed "
                        "feature is bound to a "
                        "non-categorical reference."
                    )
                )

            if observed.empty:
                continue

            families = {
                _categorical_family(
                    value
                )
                for value
                in observed.tolist()
            }

            if len(families) != 1:
                raise MLDriftEvaluatorError(
                    (
                        "Observed categorical "
                        "feature mixes incompatible "
                        "value families. "
                        f"feature={feature_name}, "
                        f"families={sorted(families)!r}"
                    )
                )

            family = next(
                iter(families)
            )

            if (
                family
                not in {
                    "boolean",
                    "text",
                    "numeric",
                }
            ):
                raise MLDriftEvaluatorError(
                    (
                        "Observed categorical "
                        "feature contains an "
                        "unsupported value family. "
                        f"feature={feature_name}, "
                        f"family={family}"
                    )
                )

            if family == "numeric":
                try:
                    values = (
                        observed.to_numpy(
                            dtype=np.float64,
                            copy=True,
                        )
                    )

                except Exception as error:
                    raise MLDriftEvaluatorError(
                        (
                            "Observed numeric-coded "
                            "categorical feature "
                            "cannot be converted to "
                            "float64."
                        )
                    ) from error

                if not (
                    np.isfinite(
                        values
                    )
                    .all()
                ):
                    raise MLDriftEvaluatorError(
                        (
                            "Observed numeric-coded "
                            "categorical feature "
                            "contains non-finite "
                            "values."
                        )
                    )

        else:
            if not isinstance(
                reference,
                MLNumericMonitoringFeatureProfile,
            ):
                raise MLDriftEvaluatorError(
                    (
                        "Numeric observed feature "
                        "is bound to a "
                        "non-numeric reference."
                    )
                )

            if (
                pd.api.types
                .is_bool_dtype(
                    series.dtype
                )
            ):
                raise MLDriftEvaluatorError(
                    (
                        "Boolean observed feature "
                        "cannot satisfy a numeric "
                        "monitoring reference."
                    )
                )

            if not (
                pd.api.types
                .is_numeric_dtype(
                    series.dtype
                )
            ):
                raise MLDriftEvaluatorError(
                    (
                        "Observed numeric feature "
                        "must have a numeric dtype. "
                        f"feature={feature_name}"
                    )
                )

            if observed.empty:
                continue

            try:
                values = (
                    observed.to_numpy(
                        dtype=np.float64,
                        copy=True,
                    )
                )

            except Exception as error:
                raise MLDriftEvaluatorError(
                    (
                        "Observed numeric feature "
                        "cannot be converted to "
                        "float64."
                    )
                ) from error

            if not (
                np.isfinite(
                    values
                )
                .all()
            ):
                raise MLDriftEvaluatorError(
                    (
                        "Observed numeric feature "
                        "contains non-finite values. "
                        f"feature={feature_name}"
                    )
                )

    return features.copy(
        deep=True
    )


# ============================================================
# NUMERIC EVALUATION
# ============================================================


def _evaluate_numeric_feature(
    *,
    series: pd.Series,
    reference: MLNumericMonitoringFeatureProfile,
) -> MLNumericDriftFeatureResult:

    total_count = int(
        len(series)
    )

    missing_mask = (
        series.isna()
    )

    missing_count = int(
        missing_mask.sum()
    )

    observed = (
        series[
            ~missing_mask
        ]
    )

    non_missing_count = int(
        len(observed)
    )

    observed_missing_rate = (
        missing_count
        /
        total_count
    )

    missing_delta = (
        observed_missing_rate
        -
        reference.missing_rate
    )

    absolute_missing_delta = abs(
        missing_delta
    )

    missingness_status = (
        ml_drift_status_for_rate_shift(
            absolute_missing_delta,
            warning_threshold=(
                ML_DRIFT_MISSING_RATE_WARNING_THRESHOLD
            ),
            drift_threshold=(
                ML_DRIFT_MISSING_RATE_DRIFT_THRESHOLD
            ),
        )
    )

    if non_missing_count == 0:
        psi = None
        distribution_status = (
            "not_evaluable"
        )

        outside_count = 0
        outside_rate = 0.0
        range_status = "ok"

        status = "drift"

    else:
        values = (
            observed.to_numpy(
                dtype=np.float64,
                copy=True,
            )
        )

        bins = np.asarray(
            [
                -np.inf,
                *reference.histogram_edges,
                np.inf,
            ],
            dtype=np.float64,
        )

        observed_counts = (
            np.histogram(
                values,
                bins=bins,
            )[0]
            .astype(
                np.int64
            )
            .tolist()
        )

        observed_rates = [
            float(
                count
                /
                non_missing_count
            )
            for count
            in observed_counts
        ]

        psi = (
            _population_stability_index(
                reference_rates=list(
                    reference.histogram_rates
                ),
                observed_rates=(
                    observed_rates
                ),
            )
        )

        distribution_status = (
            ml_drift_status_for_psi(
                psi
            )
        )

        outside_count = int(
            (
                (
                    values
                    <
                    reference.minimum
                )
                |
                (
                    values
                    >
                    reference.maximum
                )
            )
            .sum()
        )

        outside_rate = (
            outside_count
            /
            non_missing_count
        )

        range_status = (
            ml_drift_status_for_rate_shift(
                outside_rate,
                warning_threshold=(
                    ML_DRIFT_OUTSIDE_RANGE_WARNING_THRESHOLD
                ),
                drift_threshold=(
                    ML_DRIFT_OUTSIDE_RANGE_DRIFT_THRESHOLD
                ),
            )
        )

        status = (
            ml_drift_max_status(
                [
                    distribution_status,
                    missingness_status,
                    range_status,
                ]
            )
        )

    return (
        MLNumericDriftFeatureResult(
            feature_name=(
                reference.feature_name
            ),
            reference_missing_rate=(
                reference.missing_rate
            ),
            observed_total_count=(
                total_count
            ),
            observed_non_missing_count=(
                non_missing_count
            ),
            observed_missing_count=(
                missing_count
            ),
            observed_missing_rate=(
                observed_missing_rate
            ),
            missing_rate_delta=(
                missing_delta
            ),
            absolute_missing_rate_delta=(
                absolute_missing_delta
            ),
            population_stability_index=(
                psi
            ),
            distribution_status=(
                distribution_status
            ),
            missingness_status=(
                missingness_status
            ),
            outside_reference_range_count=(
                outside_count
            ),
            outside_reference_range_rate=(
                outside_rate
            ),
            range_status=(
                range_status
            ),
            status=status,
        )
    )


# ============================================================
# CATEGORICAL EVALUATION
# ============================================================


def _evaluate_categorical_feature(
    *,
    series: pd.Series,
    reference: MLCategoricalMonitoringFeatureProfile,
    model_id: str,
) -> MLCategoricalDriftFeatureResult:

    total_count = int(
        len(series)
    )

    missing_mask = (
        series.isna()
    )

    missing_count = int(
        missing_mask.sum()
    )

    observed = (
        series[
            ~missing_mask
        ]
    )

    non_missing_count = int(
        len(observed)
    )

    observed_missing_rate = (
        missing_count
        /
        total_count
    )

    missing_delta = (
        observed_missing_rate
        -
        reference.missing_rate
    )

    absolute_missing_delta = abs(
        missing_delta
    )

    missingness_status = (
        ml_drift_status_for_rate_shift(
            absolute_missing_delta,
            warning_threshold=(
                ML_DRIFT_MISSING_RATE_WARNING_THRESHOLD
            ),
            drift_threshold=(
                ML_DRIFT_MISSING_RATE_DRIFT_THRESHOLD
            ),
        )
    )

    tracked_hashes = [
        bucket.value_sha256
        for bucket
        in reference.tracked_categories
    ]

    tracked_counts = {
        digest: 0
        for digest
        in tracked_hashes
    }

    untracked_count = 0

    if non_missing_count == 0:
        psi = None
        distribution_status = (
            "not_evaluable"
        )

        observed_untracked_rate = 0.0

        status = "drift"

    else:
        for value in observed.tolist():
            try:
                digest = (
                    ml_monitoring_category_sha256(
                        model_id=model_id,
                        feature_name=(
                            reference.feature_name
                        ),
                        value=value,
                    )
                )

            except MLMonitoringProfileBuilderError as error:
                raise MLDriftEvaluatorError(
                    (
                        "Observed categorical "
                        "value could not be mapped "
                        "to the Monitoring Profile "
                        "identity domain."
                    )
                ) from error

            if digest in tracked_counts:
                tracked_counts[
                    digest
                ] += 1

            else:
                untracked_count += 1

        observed_rates = [
            (
                tracked_counts[
                    digest
                ]
                /
                non_missing_count
            )
            for digest
            in tracked_hashes
        ]

        observed_untracked_rate = (
            untracked_count
            /
            non_missing_count
        )

        observed_rates.append(
            observed_untracked_rate
        )

        reference_rates = [
            bucket.rate
            for bucket
            in reference.tracked_categories
        ]

        reference_rates.append(
            reference.other_rate
        )

        psi = (
            _population_stability_index(
                reference_rates=(
                    reference_rates
                ),
                observed_rates=(
                    observed_rates
                ),
            )
        )

        distribution_status = (
            ml_drift_status_for_psi(
                psi
            )
        )

        status = (
            ml_drift_max_status(
                [
                    distribution_status,
                    missingness_status,
                ]
            )
        )

    untracked_delta = (
        observed_untracked_rate
        -
        reference.other_rate
    )

    return (
        MLCategoricalDriftFeatureResult(
            feature_name=(
                reference.feature_name
            ),
            reference_missing_rate=(
                reference.missing_rate
            ),
            observed_total_count=(
                total_count
            ),
            observed_non_missing_count=(
                non_missing_count
            ),
            observed_missing_count=(
                missing_count
            ),
            observed_missing_rate=(
                observed_missing_rate
            ),
            missing_rate_delta=(
                missing_delta
            ),
            absolute_missing_rate_delta=(
                absolute_missing_delta
            ),
            population_stability_index=(
                psi
            ),
            distribution_status=(
                distribution_status
            ),
            missingness_status=(
                missingness_status
            ),
            reference_other_rate=(
                reference.other_rate
            ),
            observed_untracked_count=(
                untracked_count
            ),
            observed_untracked_rate=(
                observed_untracked_rate
            ),
            untracked_rate_delta=(
                untracked_delta
            ),
            absolute_untracked_rate_delta=(
                abs(
                    untracked_delta
                )
            ),
            status=status,
        )
    )


# ============================================================
# PUBLIC EVALUATOR
# ============================================================


def evaluate_ml_drift(
    *,
    observed_features: pd.DataFrame,
    observed_dataset_id: str,
    observed_preparation_session_revision: (
        int
        |
        None
    ) = None,
    monitoring_profile: MLMonitoringProfileRecord,
    model_artifact: MLModelArtifactRecord,
) -> MLDriftEvaluationRecord:
    """
    Compare an observed feature surface with the exact
    aggregate-only training reference of one trusted model.

    No persistence occurs here.

    The evaluator intentionally permits missing observed values
    even when the scoring preprocessing contract would reject
    them, because a missingness failure is itself monitoring
    evidence that must remain observable.
    """

    normalized_dataset_id = str(
        observed_dataset_id
    ).strip()

    if not normalized_dataset_id:
        raise MLDriftEvaluatorError(
            (
                "observed_dataset_id "
                "cannot be empty."
            )
        )

    normalized_observed_revision = None

    if (
        observed_preparation_session_revision
        is not None
    ):
        if isinstance(
            observed_preparation_session_revision,
            bool,
        ):
            raise MLDriftEvaluatorError(
                (
                    "observed_preparation_session_revision "
                    "must be a non-negative integer."
                )
            )

        try:
            normalized_observed_revision = int(
                observed_preparation_session_revision
            )

        except Exception as error:
            raise MLDriftEvaluatorError(
                (
                    "observed_preparation_session_revision "
                    "must be a non-negative integer."
                )
            ) from error

        if normalized_observed_revision < 0:
            raise MLDriftEvaluatorError(
                (
                    "observed_preparation_session_revision "
                    "must be a non-negative integer."
                )
            )

    artifact = (
        MLModelArtifactRecord
        .model_validate(
            model_artifact
        )
    )

    profile = (
        MLMonitoringProfileRecord
        .model_validate(
            monitoring_profile
        )
    )

    _validate_authority(
        profile=profile,
        artifact=artifact,
    )

    validated_features = (
        _validate_observed_frame(
            features=observed_features,
            artifact=artifact,
            profile=profile,
        )
    )

    feature_results = []

    profiles_by_name = {
        feature.feature_name:
            feature
        for feature
        in profile.feature_profiles
    }

    for feature_name in (
        artifact
        .training_contract
        .feature_columns
    ):
        reference = (
            profiles_by_name[
                feature_name
            ]
        )

        series = (
            validated_features[
                feature_name
            ]
        )

        if isinstance(
            reference,
            MLNumericMonitoringFeatureProfile,
        ):
            result = (
                _evaluate_numeric_feature(
                    series=series,
                    reference=reference,
                )
            )

        elif isinstance(
            reference,
            MLCategoricalMonitoringFeatureProfile,
        ):
            result = (
                _evaluate_categorical_feature(
                    series=series,
                    reference=reference,
                    model_id=(
                        artifact.model_id
                    ),
                )
            )

        else:
            raise MLDriftEvaluatorError(
                (
                    "Unsupported Monitoring "
                    "Profile feature type."
                )
            )

        feature_results.append(
            result
        )

    warning_count = sum(
        1
        for result
        in feature_results
        if result.status == "warning"
    )

    drift_count = sum(
        1
        for result
        in feature_results
        if result.status == "drift"
    )

    overall_status = (
        "drift"
        if drift_count > 0
        else
        (
            "warning"
            if warning_count > 0
            else "ok"
        )
    )

    provenance = (
        artifact.experiment_provenance
    )

    if provenance is None:
        raise MLDriftEvaluatorError(
            (
                "Experiment Provenance "
                "disappeared after authority "
                "validation."
            )
        )

    return (
        MLDriftEvaluationRecord(
            evaluation_id=(
                _new_evaluation_id()
            ),
            profile_id=(
                profile.profile_id
            ),
            model_id=(
                artifact.model_id
            ),
            workflow_id=(
                artifact.workflow_id
            ),
            reference_dataset_id=(
                profile.dataset_id
            ),
            observed_dataset_id=(
                normalized_dataset_id
            ),
            observed_preparation_session_revision=(
                normalized_observed_revision
            ),
            experiment_id=(
                provenance.experiment_id
            ),
            preparation_session_revision=(
                provenance
                .preparation_session_revision
            ),
            training_contract_sha256=(
                provenance
                .training_contract_sha256
            ),
            evaluated_at_utc=(
                _utc_now_iso()
            ),
            observed_row_count=(
                len(
                    validated_features
                )
            ),
            feature_results=(
                feature_results
            ),
            warning_feature_count=(
                warning_count
            ),
            drift_feature_count=(
                drift_count
            ),
            overall_status=(
                overall_status
            ),
        )
    )

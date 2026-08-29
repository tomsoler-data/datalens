from __future__ import annotations


import hashlib
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


from app.ml.model_artifacts import (
    MLModelArtifactRecord,
)


from app.ml.monitoring_profile import (
    MLCategoricalMonitoringBucket,
    MLCategoricalMonitoringFeatureProfile,
    MLMonitoringProfileRecord,
    MLNumericMonitoringFeatureProfile,
)


from app.ml.preprocessing import (
    MLPreprocessingRuntimeError,
    validate_ml_feature_frame,
)


# ============================================================
# VERSION
# ============================================================


ML_MONITORING_PROFILE_BUILDER_RULE_VERSION = (
    "ml_monitoring_profile_builder_v0.1"
)


# ============================================================
# POLICY
# ============================================================


ML_MONITORING_NUMERIC_QUANTILE_BUCKET_COUNT = 10


ML_MONITORING_MAX_TRACKED_CATEGORIES = 20


_CATEGORY_HASH_DOMAIN = (
    "datalens-ml-monitoring-category-v0.1"
)


# ============================================================
# ERRORS
# ============================================================


class MLMonitoringProfileBuilderError(
    RuntimeError
):
    pass


# ============================================================
# TIME / ID
# ============================================================


def _utc_now_iso(
) -> str:

    return (
        datetime.now(
            timezone.utc
        )
        .isoformat()
    )


def _new_profile_id(
) -> str:

    return (
        "monitoring-profile:"
        +
        uuid4().hex
    )


# ============================================================
# CATEGORY IDENTITY
# ============================================================


def _canonical_category_value(
    value: object,
) -> str:

    if isinstance(
        value,
        (
            bool,
            np.bool_,
        ),
    ):
        return (
            "boolean:"
            +
            (
                "true"
                if bool(
                    value
                )
                else "false"
            )
        )


    if isinstance(
        value,
        str,
    ):
        return (
            "text:"
            +
            value
        )


    if isinstance(
        value,
        (
            int,
            float,
            np.integer,
            np.floating,
        ),
    ):
        try:
            numeric = float(
                value
            )

        except Exception as error:
            raise (
                MLMonitoringProfileBuilderError(
                    (
                        "Numeric-coded categorical value "
                        "could not be normalized."
                    )
                )
            ) from error


        if not math.isfinite(
            numeric
        ):
            raise (
                MLMonitoringProfileBuilderError(
                    (
                        "Numeric-coded categorical value "
                        "must be finite."
                    )
                )
            )


        if (
            numeric
            ==
            0.0
        ):
            numeric = 0.0


        return (
            "numeric:"
            +
            numeric.hex()
        )


    raise (
        MLMonitoringProfileBuilderError(
            (
                "ML Monitoring Profile v0.1 supports "
                "categorical values from text, boolean "
                "or numeric scalar families only. "
                "Unsupported type="
                f"{type(value).__name__}"
            )
        )
    )


def ml_monitoring_category_sha256(
    *,
    model_id: str,
    feature_name: str,
    value: object,
) -> str:
    """
    Build one model-bound categorical identity.

    The raw category is never returned or persisted.

    Domain separation + model_id + feature_name prevent the
    same raw category from automatically producing the same
    identity across unrelated models/features.
    """

    normalized_model_id = str(
        model_id
    ).strip()


    normalized_feature_name = str(
        feature_name
    ).strip()


    if not normalized_model_id:
        raise (
            MLMonitoringProfileBuilderError(
                "model_id cannot be empty."
            )
        )


    if not normalized_feature_name:
        raise (
            MLMonitoringProfileBuilderError(
                "feature_name cannot be empty."
            )
        )


    canonical_value = (
        _canonical_category_value(
            value
        )
    )


    payload = (
        _CATEGORY_HASH_DOMAIN
        +
        "\x00"
        +
        normalized_model_id
        +
        "\x00"
        +
        normalized_feature_name
        +
        "\x00"
        +
        canonical_value
    )


    return (
        hashlib.sha256(
            payload.encode(
                "utf-8"
            )
        )
        .hexdigest()
    )


# ============================================================
# NUMERIC PROFILE
# ============================================================


def _numeric_histogram_edges(
    values: np.ndarray,
) -> list[
    float
]:
    """
    Build deterministic reference-quantile cut points.

    v0.1 targets ten approximately equal-frequency buckets.

    Duplicate quantiles are collapsed. Constant features
    therefore naturally produce one bucket and zero edges.
    """

    minimum = float(
        np.min(
            values
        )
    )


    maximum = float(
        np.max(
            values
        )
    )


    if (
        minimum
        ==
        maximum
    ):
        return []


    quantiles = (
        np.linspace(
            0.1,
            0.9,
            (
                ML_MONITORING_NUMERIC_QUANTILE_BUCKET_COUNT
                -
                1
            ),
        )
    )


    candidates = (
        np.quantile(
            values,
            quantiles,
        )
    )


    edges: list[
        float
    ] = []


    for raw_edge in (
        candidates.tolist()
    ):

        edge = float(
            raw_edge
        )


        if not math.isfinite(
            edge
        ):
            raise (
                MLMonitoringProfileBuilderError(
                    (
                        "Numeric monitoring quantile "
                        "produced a non-finite edge."
                    )
                )
            )


        if not (
            minimum
            <
            edge
            <
            maximum
        ):
            continue


        if (
            edges
            and
            edge
            ==
            edges[
                -1
            ]
        ):
            continue


        edges.append(
            edge
        )


    return edges


def _build_numeric_profile(
    *,
    feature_name: str,
    series: pd.Series,
) -> MLNumericMonitoringFeatureProfile:

    total_count = int(
        len(
            series
        )
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
        len(
            observed
        )
    )


    if (
        non_missing_count
        <=
        0
    ):
        raise (
            MLMonitoringProfileBuilderError(
                (
                    "Numeric monitoring feature "
                    "contains no observed value. "
                    f"feature={feature_name}"
                )
            )
        )


    try:
        values = (
            observed.to_numpy(
                dtype=np.float64,
                copy=True,
            )
        )

    except Exception as error:
        raise (
            MLMonitoringProfileBuilderError(
                (
                    "Numeric monitoring feature could "
                    "not be converted to float64. "
                    f"feature={feature_name}"
                )
            )
        ) from error


    if not (
        np.isfinite(
            values
        )
        .all()
    ):
        raise (
            MLMonitoringProfileBuilderError(
                (
                    "Numeric monitoring feature "
                    "contains non-finite values. "
                    f"feature={feature_name}"
                )
            )
        )


    q25, median, q75 = [
        float(
            value
        )

        for value
        in np.quantile(
            values,
            [
                0.25,
                0.50,
                0.75,
            ],
        )
        .tolist()
    ]


    edges = (
        _numeric_histogram_edges(
            values
        )
    )


    histogram_bins = (
        np.asarray(
            [
                -np.inf,
                *edges,
                np.inf,
            ],
            dtype=np.float64,
        )
    )


    counts = (
        np.histogram(
            values,
            bins=
                histogram_bins,
        )[
            0
        ]
        .astype(
            np.int64
        )
        .tolist()
    )


    rates = [
        float(
            count
            /
            non_missing_count
        )

        for count
        in counts
    ]


    return (
        MLNumericMonitoringFeatureProfile(
            feature_name=
                feature_name,

            total_count=
                total_count,

            non_missing_count=
                non_missing_count,

            missing_count=
                missing_count,

            missing_rate=(
                missing_count
                /
                total_count
            ),

            mean=float(
                np.mean(
                    values
                )
            ),

            std=float(
                np.std(
                    values,
                    ddof=0,
                )
            ),

            minimum=float(
                np.min(
                    values
                )
            ),

            q25=
                q25,

            median=
                median,

            q75=
                q75,

            maximum=float(
                np.max(
                    values
                )
            ),

            histogram_edges=
                edges,

            histogram_counts=[
                int(
                    count
                )
                for count
                in counts
            ],

            histogram_rates=
                rates,
        )
    )


# ============================================================
# CATEGORICAL PROFILE
# ============================================================


def _build_categorical_profile(
    *,
    model_id: str,
    feature_name: str,
    series: pd.Series,
) -> MLCategoricalMonitoringFeatureProfile:

    total_count = int(
        len(
            series
        )
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
        len(
            observed
        )
    )


    if (
        non_missing_count
        <=
        0
    ):
        raise (
            MLMonitoringProfileBuilderError(
                (
                    "Categorical monitoring feature "
                    "contains no observed category. "
                    f"feature={feature_name}"
                )
            )
        )


    counts_by_hash: dict[
        str,
        int,
    ] = {}


    canonical_by_hash: dict[
        str,
        str,
    ] = {}


    for value in (
        observed.tolist()
    ):

        canonical = (
            _canonical_category_value(
                value
            )
        )


        digest = (
            ml_monitoring_category_sha256(
                model_id=
                    model_id,

                feature_name=
                    feature_name,

                value=
                    value,
            )
        )


        previous_canonical = (
            canonical_by_hash.get(
                digest
            )
        )


        if (
            previous_canonical
            is not None
            and
            previous_canonical
            !=
            canonical
        ):
            raise (
                MLMonitoringProfileBuilderError(
                    (
                        "Categorical monitoring "
                        "SHA-256 collision detected."
                    )
                )
            )


        canonical_by_hash[
            digest
        ] = canonical


        counts_by_hash[
            digest
        ] = (
            counts_by_hash.get(
                digest,
                0,
            )
            +
            1
        )


    ordered_categories = sorted(
        counts_by_hash.items(),
        key=lambda item: (
            -item[
                1
            ],
            item[
                0
            ],
        ),
    )


    tracked = (
        ordered_categories[
            :
            ML_MONITORING_MAX_TRACKED_CATEGORIES
        ]
    )


    remaining = (
        ordered_categories[
            ML_MONITORING_MAX_TRACKED_CATEGORIES
            :
        ]
    )


    tracked_categories = [
        MLCategoricalMonitoringBucket(
            value_sha256=
                digest,

            count=
                count,

            rate=(
                count
                /
                non_missing_count
            ),
        )

        for (
            digest,
            count,
        )
        in tracked
    ]


    other_count = sum(
        count

        for (
            _,
            count,
        )
        in remaining
    )


    return (
        MLCategoricalMonitoringFeatureProfile(
            feature_name=
                feature_name,

            total_count=
                total_count,

            non_missing_count=
                non_missing_count,

            missing_count=
                missing_count,

            missing_rate=(
                missing_count
                /
                total_count
            ),

            distinct_count=
                len(
                    counts_by_hash
                ),

            tracked_categories=
                tracked_categories,

            other_count=
                other_count,

            other_rate=(
                other_count
                /
                non_missing_count
            ),
        )
    )


# ============================================================
# BUILDER
# ============================================================


def build_ml_monitoring_profile(
    *,
    x_train: pd.DataFrame,
    model_artifact: MLModelArtifactRecord,
) -> MLMonitoringProfileRecord:
    """
    Build one aggregate-only reference profile from the exact
    training split used by a trusted persisted Model Artifact.

    Authority:
        Model Artifact -> Training Contract -> x_train schema.

    No persistence occurs here.

    No target values, predictions, raw rows or raw categorical
    labels are included in the returned profile.
    """

    artifact = (
        MLModelArtifactRecord
        .model_validate(
            model_artifact
        )
    )


    provenance = (
        artifact
        .experiment_provenance
    )


    if provenance is None:
        raise (
            MLMonitoringProfileBuilderError(
                (
                    "ML Monitoring Profile requires "
                    "trusted Experiment Provenance."
                )
            )
        )


    if not isinstance(
        x_train,
        pd.DataFrame,
    ):
        raise (
            MLMonitoringProfileBuilderError(
                (
                    "ML Monitoring Profile input "
                    "must be a pandas DataFrame."
                )
            )
        )


    if x_train.empty:
        raise (
            MLMonitoringProfileBuilderError(
                (
                    "ML Monitoring Profile training "
                    "split cannot be empty."
                )
            )
        )


    if (
        len(
            x_train
        )
        !=
        artifact.train_rows
    ):
        raise (
            MLMonitoringProfileBuilderError(
                (
                    "Monitoring training split row count "
                    "does not match persisted Model "
                    "Artifact provenance."
                )
            )
        )


    contract = (
        artifact
        .training_contract
    )


    try:
        validated_features = (
            validate_ml_feature_frame(
                features=
                    x_train,

                contract=
                    contract,
            )
        )

    except MLPreprocessingRuntimeError as error:
        raise (
            MLMonitoringProfileBuilderError(
                (
                    "Monitoring training split does not "
                    "match the persisted ML Training "
                    "Contract."
                )
            )
        ) from error


    categorical_features = set(
        contract
        .categorical_feature_columns
    )


    feature_profiles = []


    for feature_name in (
        contract
        .feature_columns
    ):

        series = (
            validated_features[
                feature_name
            ]
        )


        if (
            feature_name
            in
            categorical_features
        ):
            profile = (
                _build_categorical_profile(
                    model_id=
                        artifact.model_id,

                    feature_name=
                        feature_name,

                    series=
                        series,
                )
            )

        else:
            profile = (
                _build_numeric_profile(
                    feature_name=
                        feature_name,

                    series=
                        series,
                )
            )


        feature_profiles.append(
            profile
        )


    return (
        MLMonitoringProfileRecord(
            profile_id=
                _new_profile_id(),

            model_id=
                artifact.model_id,

            workflow_id=
                artifact.workflow_id,

            dataset_id=
                artifact.dataset_id,

            experiment_id=
                provenance
                .experiment_id,

            preparation_session_revision=(
                provenance
                .preparation_session_revision
            ),

            training_contract_sha256=(
                provenance
                .training_contract_sha256
            ),

            created_at_utc=
                _utc_now_iso(),

            reference_row_count=
                int(
                    len(
                        validated_features
                    )
                ),

            feature_profiles=
                feature_profiles,
        )
    )

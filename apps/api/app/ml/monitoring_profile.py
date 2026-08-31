from __future__ import annotations


import math
import re


from typing import (
    Literal,
    Union,
)


from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# ============================================================
# VERSION
# ============================================================


ML_MONITORING_PROFILE_RULE_VERSION = (
    "ml_monitoring_profile_v0.1"
)


# ============================================================
# TYPES
# ============================================================


MLMonitoringFeatureKind = Literal[
    "numeric",
    "categorical",
]


MLMonitoringReferenceScope = Literal[
    "training_split",
]


MLMonitoringPrivacyScope = Literal[
    "aggregate_only",
]


MLMonitoringCategoricalIdentity = Literal[
    "sha256",
]


# ============================================================
# VALIDATION
# ============================================================


SHA256_PATTERN = re.compile(
    r"^[0-9a-f]{64}$"
)


MODEL_ID_PATTERN = re.compile(
    r"^model:[0-9a-f]{32}$"
)


EXPERIMENT_ID_PATTERN = re.compile(
    r"^experiment:[0-9a-f]{32}$"
)


MONITORING_PROFILE_ID_PATTERN = re.compile(
    r"^monitoring-profile:[0-9a-f]{32}$"
)


RATE_TOLERANCE = 1e-9


def _required_text(
    value: object,
    *,
    field_name: str,
) -> str:

    normalized = str(
        value
        if value is not None
        else ""
    ).strip()


    if not normalized:
        raise ValueError(
            (
                f"{field_name} "
                "cannot be empty."
            )
        )


    return normalized


def _validate_sha256(
    value: object,
    *,
    field_name: str,
) -> str:

    normalized = str(
        value
        if value is not None
        else ""
    ).strip().lower()


    if (
        SHA256_PATTERN.fullmatch(
            normalized
        )
        is None
    ):
        raise ValueError(
            (
                f"{field_name} must be a "
                "lowercase 64-character "
                "SHA-256 digest."
            )
        )


    return normalized


def _rates_close(
    left: float,
    right: float,
) -> bool:

    return math.isclose(
        left,
        right,
        rel_tol=0.0,
        abs_tol=RATE_TOLERANCE,
    )


# ============================================================
# CATEGORICAL VALUE BUCKET
# ============================================================


class MLCategoricalMonitoringBucket(
    BaseModel
):
    """
    One privacy-minimal categorical reference bucket.

    The original categorical value is deliberately absent.

    value_sha256 is the deterministic identity used later to
    compare reference and observed category distributions.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )


    value_sha256: str = Field(
        min_length=64,
        max_length=64,
    )


    count: int = Field(
        gt=0,
    )


    rate: float = Field(
        gt=0.0,
        le=1.0,
    )


    @field_validator(
        "value_sha256",
        mode="before",
    )
    @classmethod
    def validate_value_sha256(
        cls,
        value: object,
    ) -> str:

        return _validate_sha256(
            value,
            field_name=
                "value_sha256",
        )


# ============================================================
# NUMERIC FEATURE PROFILE
# ============================================================


class MLNumericMonitoringFeatureProfile(
    BaseModel
):
    """
    Aggregate-only numeric reference distribution.

    histogram_edges contains only finite internal cut points.

    histogram_counts / histogram_rates therefore contain
    len(histogram_edges) + 1 buckets.

    Raw observations are never persisted here.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )


    feature_name: str = Field(
        min_length=1,
    )


    kind: Literal[
        "numeric"
    ] = "numeric"


    total_count: int = Field(
        gt=0,
    )


    non_missing_count: int = Field(
        gt=0,
    )


    missing_count: int = Field(
        ge=0,
    )


    missing_rate: float = Field(
        ge=0.0,
        le=1.0,
    )


    mean: float


    std: float = Field(
        ge=0.0,
    )


    minimum: float


    q25: float


    median: float


    q75: float


    maximum: float


    histogram_edges: list[
        float
    ] = Field(
        default_factory=list,
        max_length=20,
    )


    histogram_counts: list[
        int
    ] = Field(
        min_length=1,
        max_length=21,
    )


    histogram_rates: list[
        float
    ] = Field(
        min_length=1,
        max_length=21,
    )


    @field_validator(
        "feature_name",
        mode="before",
    )
    @classmethod
    def normalize_feature_name(
        cls,
        value: object,
    ) -> str:

        return _required_text(
            value,
            field_name=
                "feature_name",
        )


    @field_validator(
        "histogram_counts",
    )
    @classmethod
    def validate_histogram_counts(
        cls,
        value: list[
            int
        ],
    ) -> list[
        int
    ]:

        if any(
            count < 0
            for count
            in value
        ):
            raise ValueError(
                (
                    "histogram_counts cannot "
                    "contain negative counts."
                )
            )


        return value


    @field_validator(
        "histogram_rates",
    )
    @classmethod
    def validate_histogram_rates(
        cls,
        value: list[
            float
        ],
    ) -> list[
        float
    ]:

        if any(
            (
                rate < 0.0
                or
                rate > 1.0
            )
            for rate
            in value
        ):
            raise ValueError(
                (
                    "histogram_rates must remain "
                    "between 0 and 1."
                )
            )


        return value


    @model_validator(
        mode="after"
    )
    def validate_numeric_profile(
        self,
    ) -> "MLNumericMonitoringFeatureProfile":

        if (
            self.non_missing_count
            +
            self.missing_count
            !=
            self.total_count
        ):
            raise ValueError(
                (
                    "numeric monitoring counts "
                    "must sum to total_count."
                )
            )


        expected_missing_rate = (
            self.missing_count
            /
            self.total_count
        )


        if not _rates_close(
            self.missing_rate,
            expected_missing_rate,
        ):
            raise ValueError(
                (
                    "numeric missing_rate does "
                    "not match counts."
                )
            )


        if not (
            self.minimum
            <=
            self.q25
            <=
            self.median
            <=
            self.q75
            <=
            self.maximum
        ):
            raise ValueError(
                (
                    "numeric monitoring quantiles "
                    "must be monotonically ordered."
                )
            )


        if not (
            self.minimum
            <=
            self.mean
            <=
            self.maximum
        ):
            raise ValueError(
                (
                    "numeric monitoring mean must "
                    "remain inside minimum/maximum."
                )
            )


        for (
            previous,
            current,
        ) in zip(
            self.histogram_edges,
            self.histogram_edges[
                1:
            ],
        ):
            if not (
                current
                >
                previous
            ):
                raise ValueError(
                    (
                        "histogram_edges must be "
                        "strictly increasing."
                    )
                )


        expected_bucket_count = (
            len(
                self.histogram_edges
            )
            +
            1
        )


        if (
            len(
                self.histogram_counts
            )
            !=
            expected_bucket_count
            or
            len(
                self.histogram_rates
            )
            !=
            expected_bucket_count
        ):
            raise ValueError(
                (
                    "numeric histogram shape must "
                    "equal len(edges) + 1."
                )
            )


        if (
            sum(
                self.histogram_counts
            )
            !=
            self.non_missing_count
        ):
            raise ValueError(
                (
                    "numeric histogram counts must "
                    "cover all non-missing values."
                )
            )


        if not _rates_close(
            sum(
                self.histogram_rates
            ),
            1.0,
        ):
            raise ValueError(
                (
                    "numeric histogram rates must "
                    "sum to 1."
                )
            )


        for (
            count,
            rate,
        ) in zip(
            self.histogram_counts,
            self.histogram_rates,
        ):

            expected_rate = (
                count
                /
                self.non_missing_count
            )


            if not _rates_close(
                rate,
                expected_rate,
            ):
                raise ValueError(
                    (
                        "numeric histogram rate "
                        "does not match its count."
                    )
                )


        return self


# ============================================================
# CATEGORICAL FEATURE PROFILE
# ============================================================


class MLCategoricalMonitoringFeatureProfile(
    BaseModel
):
    """
    Aggregate-only categorical reference distribution.

    Raw category labels are never stored.

    Only SHA-256 identities for tracked categories are kept.
    Remaining categories are aggregated into the `other` bucket.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )


    feature_name: str = Field(
        min_length=1,
    )


    kind: Literal[
        "categorical"
    ] = "categorical"


    category_identity: (
        MLMonitoringCategoricalIdentity
    ) = "sha256"


    total_count: int = Field(
        gt=0,
    )


    non_missing_count: int = Field(
        gt=0,
    )


    missing_count: int = Field(
        ge=0,
    )


    missing_rate: float = Field(
        ge=0.0,
        le=1.0,
    )


    distinct_count: int = Field(
        gt=0,
    )


    tracked_categories: list[
        MLCategoricalMonitoringBucket
    ] = Field(
        default_factory=list,
        max_length=50,
    )


    other_count: int = Field(
        ge=0,
    )


    other_rate: float = Field(
        ge=0.0,
        le=1.0,
    )


    @field_validator(
        "feature_name",
        mode="before",
    )
    @classmethod
    def normalize_feature_name(
        cls,
        value: object,
    ) -> str:

        return _required_text(
            value,
            field_name=
                "feature_name",
        )


    @model_validator(
        mode="after"
    )
    def validate_categorical_profile(
        self,
    ) -> "MLCategoricalMonitoringFeatureProfile":

        if (
            self.non_missing_count
            +
            self.missing_count
            !=
            self.total_count
        ):
            raise ValueError(
                (
                    "categorical monitoring counts "
                    "must sum to total_count."
                )
            )


        expected_missing_rate = (
            self.missing_count
            /
            self.total_count
        )


        if not _rates_close(
            self.missing_rate,
            expected_missing_rate,
        ):
            raise ValueError(
                (
                    "categorical missing_rate does "
                    "not match counts."
                )
            )


        hashes = [
            category.value_sha256
            for category
            in self.tracked_categories
        ]


        if (
            len(
                hashes
            )
            !=
            len(
                set(
                    hashes
                )
            )
        ):
            raise ValueError(
                (
                    "tracked categorical hashes "
                    "must be unique."
                )
            )


        tracked_count = sum(
            category.count
            for category
            in self.tracked_categories
        )


        if (
            tracked_count
            +
            self.other_count
            !=
            self.non_missing_count
        ):
            raise ValueError(
                (
                    "categorical buckets must cover "
                    "all non-missing values."
                )
            )


        tracked_rate = sum(
            category.rate
            for category
            in self.tracked_categories
        )


        if not _rates_close(
            tracked_rate
            +
            self.other_rate,
            1.0,
        ):
            raise ValueError(
                (
                    "categorical bucket rates must "
                    "sum to 1."
                )
            )


        for category in (
            self.tracked_categories
        ):

            expected_rate = (
                category.count
                /
                self.non_missing_count
            )


            if not _rates_close(
                category.rate,
                expected_rate,
            ):
                raise ValueError(
                    (
                        "categorical tracked rate "
                        "does not match its count."
                    )
                )


        expected_other_rate = (
            self.other_count
            /
            self.non_missing_count
        )


        if not _rates_close(
            self.other_rate,
            expected_other_rate,
        ):
            raise ValueError(
                (
                    "categorical other_rate does "
                    "not match other_count."
                )
            )


        minimum_distinct_count = (
            len(
                self.tracked_categories
            )
            +
            (
                1
                if self.other_count
                >
                0
                else 0
            )
        )


        if (
            self.distinct_count
            <
            minimum_distinct_count
        ):
            raise ValueError(
                (
                    "distinct_count is smaller "
                    "than represented category "
                    "buckets."
                )
            )


        if (
            self.other_count
            ==
            0
            and
            self.distinct_count
            !=
            len(
                self.tracked_categories
            )
        ):
            raise ValueError(
                (
                    "without an other bucket, "
                    "distinct_count must equal "
                    "tracked category count."
                )
            )


        return self


# ============================================================
# FEATURE UNION
# ============================================================


MLMonitoringFeatureProfile = Union[
    MLNumericMonitoringFeatureProfile,
    MLCategoricalMonitoringFeatureProfile,
]


# ============================================================
# MONITORING PROFILE
# ============================================================


class MLMonitoringProfileRecord(
    BaseModel
):
    """
    Durable privacy-minimal training reference for one trusted
    Model Artifact.

    The reference is based only on the training split.

    It intentionally contains no:
    - raw dataset rows;
    - target observations;
    - predictions;
    - raw categorical labels;
    - fitted estimator bytes;
    - model filesystem paths;
    - secrets.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )


    profile_id: str = Field(
        min_length=1,
    )


    model_id: str = Field(
        min_length=1,
    )


    workflow_id: str = Field(
        min_length=1,
    )


    dataset_id: str = Field(
        min_length=1,
    )


    experiment_id: str = Field(
        min_length=1,
    )


    preparation_session_revision: int = Field(
        ge=0,
    )


    training_contract_sha256: str = Field(
        min_length=64,
        max_length=64,
    )


    created_at_utc: str = Field(
        min_length=1,
    )


    reference_scope: (
        MLMonitoringReferenceScope
    ) = "training_split"


    reference_row_count: int = Field(
        gt=0,
    )


    feature_profiles: list[
        MLMonitoringFeatureProfile
    ] = Field(
        min_length=1,
        max_length=256,
    )


    privacy_scope: (
        MLMonitoringPrivacyScope
    ) = "aggregate_only"


    categorical_identity: (
        MLMonitoringCategoricalIdentity
    ) = "sha256"


    rule_version: Literal[
        "ml_monitoring_profile_v0.1"
    ] = ML_MONITORING_PROFILE_RULE_VERSION


    @field_validator(
        "profile_id",
        "model_id",
        "workflow_id",
        "dataset_id",
        "experiment_id",
        "created_at_utc",
        mode="before",
    )
    @classmethod
    def normalize_required_text(
        cls,
        value: object,
        info,
    ) -> str:

        return _required_text(
            value,
            field_name=
                info.field_name,
        )


    @field_validator(
        "profile_id",
    )
    @classmethod
    def validate_profile_id(
        cls,
        value: str,
    ) -> str:

        if (
            MONITORING_PROFILE_ID_PATTERN
            .fullmatch(
                value
            )
            is None
        ):
            raise ValueError(
                (
                    "profile_id must be a "
                    "server-shaped monitoring "
                    "profile identifier."
                )
            )


        return value


    @field_validator(
        "model_id",
    )
    @classmethod
    def validate_model_id(
        cls,
        value: str,
    ) -> str:

        if (
            MODEL_ID_PATTERN.fullmatch(
                value
            )
            is None
        ):
            raise ValueError(
                (
                    "model_id must be a "
                    "server-shaped model "
                    "identifier."
                )
            )


        return value


    @field_validator(
        "experiment_id",
    )
    @classmethod
    def validate_experiment_id(
        cls,
        value: str,
    ) -> str:

        if (
            EXPERIMENT_ID_PATTERN
            .fullmatch(
                value
            )
            is None
        ):
            raise ValueError(
                (
                    "experiment_id must be a "
                    "server-shaped experiment "
                    "identifier."
                )
            )


        return value


    @field_validator(
        "training_contract_sha256",
        mode="before",
    )
    @classmethod
    def validate_training_contract_sha256(
        cls,
        value: object,
    ) -> str:

        return _validate_sha256(
            value,
            field_name=
                "training_contract_sha256",
        )


    @model_validator(
        mode="after"
    )
    def validate_monitoring_profile(
        self,
    ) -> "MLMonitoringProfileRecord":

        feature_names = [
            feature.feature_name
            for feature
            in self.feature_profiles
        ]


        if (
            len(
                feature_names
            )
            !=
            len(
                set(
                    feature_names
                )
            )
        ):
            raise ValueError(
                (
                    "monitoring feature names "
                    "must be unique."
                )
            )


        for feature in (
            self.feature_profiles
        ):
            if (
                feature.total_count
                !=
                self.reference_row_count
            ):
                raise ValueError(
                    (
                        "every feature profile "
                        "must describe the complete "
                        "training reference row "
                        "surface."
                    )
                )


        return self

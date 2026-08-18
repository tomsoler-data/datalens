from __future__ import annotations


import re
import unicodedata


from typing import (
    Any,
    Literal,
)


import pandas as pd


from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


from app.profiling.types import (
    infer_analytical_type,
)


# ============================================================
# VERSION
# ============================================================


ENTITY_OUTLIER_RULE_VERSION = (
    "entity_outlier_engine_v0.1"
)


# ============================================================
# CONSTANTS
# ============================================================


MIN_VALID_ENTITY_OBSERVATIONS = 8

DEFAULT_TOP_ENTITY_LIMIT = 25


# These are behavioural measures, not contextual descriptors.
#
# The names are generic enough to support the current
# customer-behaviour analytical view while still allowing the
# fallback logic below to work for other entity views.
CUSTOMER_BEHAVIOUR_PRIORITY = (
    "total_spend",
    "purchase_sessions",
    "total_items",
    "average_basket",
    "median_basket",
    "average_items_per_basket",
)


CONTEXTUAL_METRIC_SIGNALS = {
    "age",
    "birth",
    "year",
    "date",
    "time",
    "timestamp",
}


# ============================================================
# TYPES
# ============================================================


OutlierDirection = Literal[
    "low",
    "high",
]


class EntityOutlierMetricThreshold(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    metric: str = Field(
        min_length=1
    )

    valid_observations: int = Field(
        ge=0
    )

    q1: float

    q3: float

    iqr: float

    lower_bound: float

    upper_bound: float

    flagged_count: int = Field(
        ge=0
    )


class EntityOutlierEvidence(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    metric: str = Field(
        min_length=1
    )

    value: float

    direction: OutlierDirection

    q1: float

    q3: float

    iqr: float

    lower_bound: float

    upper_bound: float

    distance_iqr: float = Field(
        ge=0.0
    )

    score: float = Field(
        ge=0.0
    )


class EntityOutlierCandidate(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    entity: str = Field(
        min_length=1
    )

    anomaly_score: float = Field(
        ge=0.0
    )

    outlier_metric_count: int = Field(
        ge=1
    )

    evidence: list[
        EntityOutlierEvidence
    ] = Field(
        min_length=1
    )


class EntityOutlierViewResult(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    dataset_id: str = Field(
        min_length=1
    )

    dataset_filename: str = Field(
        min_length=1
    )

    derivation_type: str

    operation: str

    entity_column: str = Field(
        min_length=1
    )

    entity_count: int = Field(
        ge=0
    )

    evaluated_metrics: list[
        str
    ] = Field(
        default_factory=list
    )

    primary_metric: (
        str
        | None
    ) = None

    thresholds: list[
        EntityOutlierMetricThreshold
    ] = Field(
        default_factory=list
    )

    flagged_entity_count: int = Field(
        ge=0
    )

    top_entities: list[
        EntityOutlierCandidate
    ] = Field(
        default_factory=list
    )

    notes: list[
        str
    ] = Field(
        default_factory=list
    )


class EntityOutlierReport(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    status: Literal[
        "ready"
    ] = "ready"

    candidate_view_count: int = Field(
        ge=0
    )

    evaluated_view_count: int = Field(
        ge=0
    )

    total_flagged_entity_count: int = Field(
        ge=0
    )

    results: list[
        EntityOutlierViewResult
    ] = Field(
        default_factory=list
    )

    notes: list[
        str
    ] = Field(
        default_factory=list
    )

    rule_version: str = (
        ENTITY_OUTLIER_RULE_VERSION
    )


# ============================================================
# TEXT HELPERS
# ============================================================


def _normalize_text(
    value: Any,
) -> str:
    text = (
        unicodedata.normalize(
            "NFKD",
            str(
                value
            ),
        )
        .encode(
            "ascii",
            "ignore",
        )
        .decode(
            "ascii"
        )
        .casefold()
    )


    return re.sub(
        r"[^a-z0-9]+",
        "_",
        text,
    ).strip(
        "_"
    )


def _record_text(
    record: dict[
        str,
        Any,
    ],
    key: str,
) -> str:
    return str(
        record.get(
            key
        )
        or
        ""
    ).strip()


# ============================================================
# VIEW SELECTION
# ============================================================


def _is_entity_analytical_view(
    record: dict[
        str,
        Any,
    ],
) -> bool:
    if not bool(
        record.get(
            "is_derived",
            False,
        )
    ):
        return False


    derivation_type = (
        _record_text(
            record,
            "derivation_type",
        )
    )


    provenance = (
        record.get(
            "provenance"
        )
        or {}
    )


    operation = str(
        provenance.get(
            "operation"
        )
        or
        ""
    ).strip()


    if (
        operation
        ==
        "customer_behavior_materialization"
    ):
        return True


    if (
        derivation_type
        ==
        "entity_additive_measure"
    ):
        return True


    return False


# ============================================================
# ENTITY RESOLUTION
# ============================================================


def _entity_column_from_provenance(
    record: dict[
        str,
        Any,
    ],
) -> (
    str
    | None
):
    provenance = (
        record.get(
            "provenance"
        )
        or {}
    )


    entity_column = str(
        provenance.get(
            "entity_column"
        )
        or
        ""
    ).strip()


    if entity_column:
        return (
            entity_column
        )


    grain = str(
        provenance.get(
            "grain"
        )
        or
        ""
    ).strip()


    if grain:
        return (
            grain
        )


    return None


def _infer_entity_column(
    dataframe: pd.DataFrame,
) -> (
    str
    | None
):
    candidates: list[
        str
    ] = []


    for raw_column in (
        dataframe.columns
    ):
        column = str(
            raw_column
        )


        analytical_type = (
            infer_analytical_type(
                column,
                dataframe[
                    raw_column
                ],
            )
        )


        if (
            analytical_type.get(
                "type"
            )
            ==
            "identifier"
        ):
            candidates.append(
                column
            )


    if (
        len(
            candidates
        )
        !=
        1
    ):
        return None


    return (
        candidates[
            0
        ]
    )


def _resolve_entity_column(
    record: dict[
        str,
        Any,
    ],
    dataframe: pd.DataFrame,
) -> (
    str
    | None
):
    entity_column = (
        _entity_column_from_provenance(
            record
        )
    )


    if (
        entity_column
        and
        entity_column
        in dataframe.columns
    ):
        return (
            entity_column
        )


    return (
        _infer_entity_column(
            dataframe
        )
    )


# ============================================================
# METRIC SELECTION
# ============================================================


def _looks_contextual(
    column: str,
) -> bool:
    normalized = (
        _normalize_text(
            column
        )
    )


    tokens = {
        token

        for token
        in normalized.split(
            "_"
        )

        if token
    }


    return bool(
        tokens
        &
        CONTEXTUAL_METRIC_SIGNALS
    )


def _is_quantitative_metric(
    *,
    dataframe: pd.DataFrame,
    column: str,
) -> bool:
    if (
        column
        not in dataframe.columns
    ):
        return False


    if (
        _looks_contextual(
            column
        )
    ):
        return False


    analytical_type = (
        infer_analytical_type(
            column,
            dataframe[
                column
            ],
        )
    )


    return (
        analytical_type.get(
            "type"
        )
        ==
        "quantitative"
    )


def _customer_behaviour_metrics(
    dataframe: pd.DataFrame,
) -> list[
    str
]:
    metrics: list[
        str
    ] = []


    for metric in (
        CUSTOMER_BEHAVIOUR_PRIORITY
    ):
        if (
            metric
            in dataframe.columns
            and
            _is_quantitative_metric(
                dataframe=
                    dataframe,
                column=
                    metric,
            )
        ):
            metrics.append(
                metric
            )


    return (
        metrics
    )


def _generic_entity_metrics(
    *,
    record: dict[
        str,
        Any,
    ],
    dataframe: pd.DataFrame,
    entity_column: str,
) -> list[
    str
]:
    provenance = (
        record.get(
            "provenance"
        )
        or {}
    )


    preferred: list[
        str
    ] = []


    target_measure = str(
        provenance.get(
            "target_measure_column"
        )
        or
        ""
    ).strip()


    if target_measure:
        preferred.append(
            target_measure
        )


    if (
        "event_count"
        in dataframe.columns
    ):
        preferred.append(
            "event_count"
        )


    metrics: list[
        str
    ] = []


    for column in [
        *preferred,
        *[
            str(
                raw_column
            )

            for raw_column
            in dataframe.columns
        ],
    ]:
        if (
            not column
            or
            column
            ==
            entity_column
            or
            column
            in metrics
        ):
            continue


        if (
            _is_quantitative_metric(
                dataframe=
                    dataframe,
                column=
                    column,
            )
        ):
            metrics.append(
                column
            )


    return (
        metrics
    )


def _resolve_metrics(
    *,
    record: dict[
        str,
        Any,
    ],
    dataframe: pd.DataFrame,
    entity_column: str,
) -> list[
    str
]:
    provenance = (
        record.get(
            "provenance"
        )
        or {}
    )


    operation = str(
        provenance.get(
            "operation"
        )
        or
        ""
    ).strip()


    if (
        operation
        ==
        "customer_behavior_materialization"
    ):
        return (
            _customer_behaviour_metrics(
                dataframe
            )
        )


    return (
        _generic_entity_metrics(
            record=
                record,
            dataframe=
                dataframe,
            entity_column=
                entity_column,
        )
    )


def _primary_metric(
    metrics: list[
        str
    ],
) -> (
    str
    | None
):
    for metric in (
        CUSTOMER_BEHAVIOUR_PRIORITY
    ):
        if (
            metric
            in metrics
        ):
            return (
                metric
            )


    if metrics:
        return (
            metrics[
                0
            ]
        )


    return None


# ============================================================
# IQR THRESHOLD
# ============================================================


def _build_threshold(
    *,
    dataframe: pd.DataFrame,
    metric: str,
) -> (
    EntityOutlierMetricThreshold
    | None
):
    numeric = (
        pd.to_numeric(
            dataframe[
                metric
            ],
            errors=
                "coerce",
        )
        .dropna()
    )


    valid_observations = int(
        len(
            numeric
        )
    )


    if (
        valid_observations
        <
        MIN_VALID_ENTITY_OBSERVATIONS
    ):
        return None


    if (
        int(
            numeric.nunique()
        )
        <=
        1
    ):
        return None


    q1 = float(
        numeric.quantile(
            0.25
        )
    )


    q3 = float(
        numeric.quantile(
            0.75
        )
    )


    iqr = float(
        q3
        -
        q1
    )


    if (
        iqr
        <=
        0.0
    ):
        return None


    lower_bound = float(
        q1
        -
        (
            1.5
            *
            iqr
        )
    )


    upper_bound = float(
        q3
        +
        (
            1.5
            *
            iqr
        )
    )


    flagged_count = int(
        (
            (
                numeric
                <
                lower_bound
            )
            |
            (
                numeric
                >
                upper_bound
            )
        )
        .sum()
    )


    return (
        EntityOutlierMetricThreshold(
            metric=
                metric,

            valid_observations=
                valid_observations,

            q1=
                q1,

            q3=
                q3,

            iqr=
                iqr,

            lower_bound=
                lower_bound,

            upper_bound=
                upper_bound,

            flagged_count=
                flagged_count,
        )
    )


# ============================================================
# EVIDENCE
# ============================================================


def _evidence_for_value(
    *,
    metric: str,
    value: float,
    threshold: EntityOutlierMetricThreshold,
) -> (
    EntityOutlierEvidence
    | None
):
    if (
        value
        <
        threshold.lower_bound
    ):
        distance_iqr = float(
            (
                threshold.lower_bound
                -
                value
            )
            /
            threshold.iqr
        )


        direction: OutlierDirection = (
            "low"
        )


    elif (
        value
        >
        threshold.upper_bound
    ):
        distance_iqr = float(
            (
                value
                -
                threshold.upper_bound
            )
            /
            threshold.iqr
        )


        direction = (
            "high"
        )


    else:
        return None


    # Every threshold crossing receives a base score of 1.
    # Additional distance beyond the IQR boundary increases
    # the score continuously.
    score = float(
        1.0
        +
        distance_iqr
    )


    return (
        EntityOutlierEvidence(
            metric=
                metric,

            value=
                value,

            direction=
                direction,

            q1=
                threshold.q1,

            q3=
                threshold.q3,

            iqr=
                threshold.iqr,

            lower_bound=
                threshold.lower_bound,

            upper_bound=
                threshold.upper_bound,

            distance_iqr=
                distance_iqr,

            score=
                score,
        )
    )


# ============================================================
# SINGLE VIEW
# ============================================================


def _evaluate_entity_view(
    record: dict[
        str,
        Any,
    ],
    *,
    top_limit: int,
) -> (
    EntityOutlierViewResult
    | None
):
    dataframe = (
        record.get(
            "dataframe"
        )
    )


    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        return None


    if dataframe.empty:
        return None


    entity_column = (
        _resolve_entity_column(
            record,
            dataframe,
        )
    )


    if (
        entity_column
        is None
    ):
        return None


    if (
        dataframe[
            entity_column
        ]
        .dropna()
        .duplicated()
        .any()
    ):
        # Entity-outlier analysis must operate at one row per
        # entity. If the view does not respect that grain,
        # DataLens abstains rather than silently aggregate again.
        return None


    metrics = (
        _resolve_metrics(
            record=
                record,
            dataframe=
                dataframe,
            entity_column=
                entity_column,
        )
    )


    if not metrics:
        return None


    thresholds: list[
        EntityOutlierMetricThreshold
    ] = []


    for metric in (
        metrics
    ):
        threshold = (
            _build_threshold(
                dataframe=
                    dataframe,
                metric=
                    metric,
            )
        )


        if (
            threshold
            is not None
        ):
            thresholds.append(
                threshold
            )


    if not thresholds:
        return None


    evidence_by_entity: dict[
        str,
        list[
            EntityOutlierEvidence
        ],
    ] = {}


    for threshold in (
        thresholds
    ):
        numeric = (
            pd.to_numeric(
                dataframe[
                    threshold.metric
                ],
                errors=
                    "coerce",
            )
        )


        for index in (
            dataframe.index
        ):
            raw_entity = (
                dataframe.at[
                    index,
                    entity_column,
                ]
            )


            if pd.isna(
                raw_entity
            ):
                continue


            raw_value = (
                numeric.at[
                    index
                ]
            )


            if pd.isna(
                raw_value
            ):
                continue


            evidence = (
                _evidence_for_value(
                    metric=
                        threshold.metric,

                    value=
                        float(
                            raw_value
                        ),

                    threshold=
                        threshold,
                )
            )


            if (
                evidence
                is None
            ):
                continue


            entity = str(
                raw_entity
            )


            evidence_by_entity.setdefault(
                entity,
                [],
            ).append(
                evidence
            )


    candidates: list[
        EntityOutlierCandidate
    ] = []


    for (
        entity,
        evidence,
    ) in (
        evidence_by_entity
        .items()
    ):
        evidence = sorted(
            evidence,

            key=lambda item:
                item.score,

            reverse=True,
        )


        anomaly_score = float(
            sum(
                item.score

                for item
                in evidence
            )
        )


        candidates.append(
            EntityOutlierCandidate(
                entity=
                    entity,

                anomaly_score=
                    anomaly_score,

                outlier_metric_count=
                    len(
                        evidence
                    ),

                evidence=
                    evidence,
            )
        )


    candidates.sort(
        key=lambda candidate: (
            candidate.anomaly_score,
            candidate.outlier_metric_count,
        ),
        reverse=True,
    )


    provenance = (
        record.get(
            "provenance"
        )
        or {}
    )


    derivation_type = (
        _record_text(
            record,
            "derivation_type",
        )
    )


    operation = str(
        provenance.get(
            "operation"
        )
        or
        ""
    ).strip()


    entity_count = int(
        dataframe[
            entity_column
        ]
        .nunique(
            dropna=True
        )
    )


    notes = [
        (
            "Les valeurs atypiques sont détectées "
            "au grain de l'entité, jamais directement "
            "sur les lignes transactionnelles."
        ),

        (
            "La règle IQR 1,5× est appliquée "
            "séparément à chaque métrique "
            "comportementale quantitative."
        ),

        (
            "Une entité peut être atypique sur une "
            "ou plusieurs métriques. Le score agrège "
            "la distance aux bornes IQR."
        ),

        (
            "Aucune entité n'est supprimée ou exclue "
            "automatiquement."
        ),
    ]


    return (
        EntityOutlierViewResult(
            dataset_id=
                _record_text(
                    record,
                    "dataset_id",
                ),

            dataset_filename=
                _record_text(
                    record,
                    "filename",
                ),

            derivation_type=
                derivation_type,

            operation=
                operation,

            entity_column=
                entity_column,

            entity_count=
                entity_count,

            evaluated_metrics=[
                threshold.metric

                for threshold
                in thresholds
            ],

            primary_metric=
                _primary_metric(
                    [
                        threshold.metric

                        for threshold
                        in thresholds
                    ]
                ),

            thresholds=
                thresholds,

            flagged_entity_count=
                len(
                    candidates
                ),

            top_entities=
                candidates[
                    :top_limit
                ],

            notes=
                notes,
        )
    )


# ============================================================
# PUBLIC API
# ============================================================


def detect_entity_outliers(
    *,
    datasets: list[
        dict[
            str,
            Any,
        ]
    ],

    top_limit: int = (
        DEFAULT_TOP_ENTITY_LIMIT
    ),
) -> EntityOutlierReport:
    """
    Detect atypical entities from safe analytical views.

    This engine deliberately does NOT aggregate raw event rows
    and does NOT create joins.

    It consumes analytical views already materialized by
    DataLens at a validated entity grain.

    Current supported views:
    - customer behaviour views;
    - generic entity additive-measure views.

    Therefore:

        raw fact rows
            -> ignored

        safe derived customer/entity view
            -> eligible
    """

    if (
        top_limit
        <
        1
    ):
        raise ValueError(
            "top_limit must be greater than zero."
        )


    candidate_records = [
        record

        for record
        in datasets

        if (
            _is_entity_analytical_view(
                record
            )
        )
    ]


    results: list[
        EntityOutlierViewResult
    ] = []


    for record in (
        candidate_records
    ):
        result = (
            _evaluate_entity_view(
                record,

                top_limit=
                    top_limit,
            )
        )


        if (
            result
            is not None
        ):
            results.append(
                result
            )


    total_flagged = sum(
        result.flagged_entity_count

        for result
        in results
    )


    notes = [
        (
            "Entity-outlier detection consumes only "
            "validated derived analytical views."
        ),

        (
            "Raw transaction rows are never treated "
            "as one entity observation."
        ),

        (
            "Contextual variables such as age, birth "
            "year and dates are excluded from the "
            "behavioural anomaly score."
        ),

        (
            "Outlier flags are analytical signals, "
            "not automatic fraud labels or deletion rules."
        ),
    ]


    return (
        EntityOutlierReport(
            candidate_view_count=
                len(
                    candidate_records
                ),

            evaluated_view_count=
                len(
                    results
                ),

            total_flagged_entity_count=
                total_flagged,

            results=
                results,

            notes=
                notes,
        )
    )
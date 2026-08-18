from __future__ import annotations


from collections import (
    defaultdict,
)

from typing import (
    Literal,
)


from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


from app.analysis.entity_outliers import (
    EntityOutlierCandidate,
    EntityOutlierEvidence,
    EntityOutlierReport,
)


# ============================================================
# VERSION
# ============================================================


ENTITY_OUTLIER_PROFILE_RULE_VERSION = (
    "entity_outlier_profile_v0.1"
)


# ============================================================
# THRESHOLDS
# ============================================================
#
# IMPORTANT
#
# These thresholds do NOT define whether an observation is
# statistically outside an IQR boundary.
#
# That decision has already been made by:
#
#     entity_outlier_engine_v0.1
#
# This layer only decides how to PRESENT and PRIORITISE the
# resulting signals.
#
# A priority profile must:
#
# - be atypical on at least two behavioural metrics;
# - contain at least one extremely distant IQR signal.
#
# The current extreme threshold is deliberately conservative.
#
# On the real Lapage dataset:
#
#     c_1609
#     c_4958
#     c_3454
#     c_6714
#
# are hundreds of IQR units beyond at least one threshold,
# while the fifth ranked client is only around 11 IQR units
# beyond its strongest threshold.
#
# ============================================================


DEFAULT_PRIORITY_MIN_SIGNAL_COUNT = 2

DEFAULT_PRIORITY_MIN_MAX_DISTANCE_IQR = 20.0

DEFAULT_STRONG_MIN_MAX_DISTANCE_IQR = 5.0

DEFAULT_PROFILE_TOP_LIMIT = 25


# ============================================================
# METRIC FAMILIES
# ============================================================


CUSTOMER_BEHAVIOUR_METRIC_FAMILIES = {
    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    "total_spend":
        "volume",

    "purchase_sessions":
        "volume",

    "total_items":
        "volume",


    # --------------------------------------------------------
    # BASKET
    # --------------------------------------------------------

    "average_basket":
        "basket",

    "median_basket":
        "basket",


    # --------------------------------------------------------
    # INTENSITY
    # --------------------------------------------------------

    "average_items_per_basket":
        "intensity",
}


# ============================================================
# TYPES
# ============================================================


MetricFamily = Literal[
    "volume",
    "basket",
    "intensity",
    "other",
]


ProfileSeverity = Literal[
    "extreme",
    "strong",
    "moderate",
]


ProfileKind = Literal[
    "priority_profile",
    "behavioral_signal",
]


class EntityOutlierProfileEvidence(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    metric: str = Field(
        min_length=1
    )

    family: MetricFamily

    value: float

    direction: Literal[
        "low",
        "high",
    ]

    distance_iqr: float = Field(
        ge=0.0
    )

    score: float = Field(
        ge=0.0
    )


class EntityOutlierProfile(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    entity: str = Field(
        min_length=1
    )

    profile_kind: ProfileKind

    severity: ProfileSeverity

    anomaly_score: float = Field(
        ge=0.0
    )

    signal_count: int = Field(
        ge=1
    )

    signal_families: list[
        MetricFamily
    ] = Field(
        min_length=1
    )

    dominant_family: MetricFamily

    max_distance_iqr: float = Field(
        ge=0.0
    )

    title: str = Field(
        min_length=1
    )

    rationale: str = Field(
        min_length=1
    )

    evidence: list[
        EntityOutlierProfileEvidence
    ] = Field(
        min_length=1
    )


class EntityOutlierProfileViewResult(
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

    entity_column: str = Field(
        min_length=1
    )

    entity_count: int = Field(
        ge=0
    )

    source_flagged_entity_count: int = Field(
        ge=0
    )

    classified_entity_count: int = Field(
        ge=0
    )

    priority_profile_count: int = Field(
        ge=0
    )

    behavioral_signal_count: int = Field(
        ge=0
    )

    unclassified_flagged_entity_count: int = Field(
        ge=0
    )

    priority_profiles: list[
        EntityOutlierProfile
    ] = Field(
        default_factory=list
    )

    behavioral_signals: list[
        EntityOutlierProfile
    ] = Field(
        default_factory=list
    )


class EntityOutlierProfileReport(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    status: Literal[
        "ready"
    ] = "ready"

    source_view_count: int = Field(
        ge=0
    )

    classified_view_count: int = Field(
        ge=0
    )

    source_flagged_entity_count: int = Field(
        ge=0
    )

    classified_entity_count: int = Field(
        ge=0
    )

    priority_profile_count: int = Field(
        ge=0
    )

    behavioral_signal_count: int = Field(
        ge=0
    )

    unclassified_flagged_entity_count: int = Field(
        ge=0
    )

    results: list[
        EntityOutlierProfileViewResult
    ] = Field(
        default_factory=list
    )

    notes: list[
        str
    ] = Field(
        default_factory=list
    )

    rule_version: str = (
        ENTITY_OUTLIER_PROFILE_RULE_VERSION
    )


# ============================================================
# METRIC FAMILY
# ============================================================


def metric_family(
    metric: str,
) -> MetricFamily:
    value = (
        CUSTOMER_BEHAVIOUR_METRIC_FAMILIES
        .get(
            metric
        )
    )


    if (
        value
        ==
        "volume"
    ):
        return "volume"


    if (
        value
        ==
        "basket"
    ):
        return "basket"


    if (
        value
        ==
        "intensity"
    ):
        return "intensity"


    return "other"


# ============================================================
# FAMILY LABELS
# ============================================================


def _family_label(
    family: MetricFamily,
) -> str:
    if (
        family
        ==
        "volume"
    ):
        return (
            "volume d’activité"
        )


    if (
        family
        ==
        "basket"
    ):
        return (
            "panier"
        )


    if (
        family
        ==
        "intensity"
    ):
        return (
            "intensité d’achat"
        )


    return (
        "comportement"
    )


# ============================================================
# EVIDENCE CONVERSION
# ============================================================


def _profile_evidence(
    evidence:
        EntityOutlierEvidence,
) -> EntityOutlierProfileEvidence:
    return (
        EntityOutlierProfileEvidence(
            metric=
                evidence.metric,

            family=
                metric_family(
                    evidence.metric
                ),

            value=
                evidence.value,

            direction=
                evidence.direction,

            distance_iqr=
                evidence.distance_iqr,

            score=
                evidence.score,
        )
    )


# ============================================================
# DOMINANT FAMILY
# ============================================================


def _dominant_family(
    evidence: list[
        EntityOutlierProfileEvidence
    ],
) -> MetricFamily:
    counts: dict[
        MetricFamily,
        int,
    ] = defaultdict(
        int
    )


    distances: dict[
        MetricFamily,
        float,
    ] = defaultdict(
        float
    )


    for item in evidence:
        counts[
            item.family
        ] += 1


        distances[
            item.family
        ] += (
            item.distance_iqr
        )


    families = list(
        counts.keys()
    )


    if not families:
        return "other"


    families.sort(
        key=lambda family: (
            counts[
                family
            ],

            distances[
                family
            ],
        ),
        reverse=True,
    )


    return (
        families[
            0
        ]
    )


# ============================================================
# SEVERITY
# ============================================================


def _severity(
    *,
    max_distance_iqr: float,
    signal_count: int,
    priority_min_signal_count: int,
    priority_min_max_distance_iqr: float,
    strong_min_max_distance_iqr: float,
) -> ProfileSeverity:
    if (
        signal_count
        >=
        priority_min_signal_count

        and

        max_distance_iqr
        >=
        priority_min_max_distance_iqr
    ):
        return "extreme"


    if (
        max_distance_iqr
        >=
        strong_min_max_distance_iqr

        or

        signal_count
        >=
        3
    ):
        return "strong"


    return "moderate"


# ============================================================
# PROFILE KIND
# ============================================================


def _profile_kind(
    *,
    severity: ProfileSeverity,
) -> ProfileKind:
    if (
        severity
        ==
        "extreme"
    ):
        return (
            "priority_profile"
        )


    return (
        "behavioral_signal"
    )


# ============================================================
# TITLE
# ============================================================


def _profile_title(
    *,
    profile_kind:
        ProfileKind,
    dominant_family:
        MetricFamily,
) -> str:
    family = (
        _family_label(
            dominant_family
        )
    )


    if (
        profile_kind
        ==
        "priority_profile"
    ):
        return (
            f"Profil extrême · {family}"
        )


    return (
        f"Signal atypique · {family}"
    )


# ============================================================
# RATIONALE
# ============================================================


def _profile_rationale(
    *,
    profile_kind:
        ProfileKind,
    severity:
        ProfileSeverity,
    signal_count:
        int,
    signal_families:
        list[
            MetricFamily
        ],
    dominant_family:
        MetricFamily,
    max_distance_iqr:
        float,
) -> str:
    family_label = (
        _family_label(
            dominant_family
        )
    )


    family_count = len(
        signal_families
    )


    if (
        profile_kind
        ==
        "priority_profile"
    ):
        return (
            f"{signal_count} métrique(s) comportementale(s) "
            f"sont atypiques. Le signal le plus extrême "
            f"dépasse sa borne IQR de "
            f"{max_distance_iqr:.1f} IQR. "
            f"Le profil est principalement associé au "
            f"{family_label}. "
            f"Une revue prioritaire est recommandée, "
            f"sans conclure automatiquement à une fraude, "
            f"à un statut BtoB ou à une erreur de données."
        )


    if (
        severity
        ==
        "strong"
    ):
        return (
            f"{signal_count} métrique(s) présentent une "
            f"atypicité marquée, principalement sur le "
            f"{family_label}. "
            f"Le signal le plus éloigné dépasse sa borne "
            f"IQR de {max_distance_iqr:.1f} IQR. "
            f"Ce profil mérite une vérification, mais "
            f"n'est pas classé parmi les profils extrêmes."
        )


    if (
        family_count
        >
        1
    ):
        return (
            f"{signal_count} métrique(s) présentent une "
            f"atypicité répartie sur {family_count} "
            f"dimensions comportementales. "
            f"Le signal le plus éloigné dépasse sa borne "
            f"IQR de {max_distance_iqr:.1f} IQR. "
            f"Le signal est conservé pour exploration."
        )


    return (
        f"{signal_count} métrique(s) présentent une "
        f"atypicité localisée sur le {family_label}. "
        f"Le signal le plus éloigné dépasse sa borne "
        f"IQR de {max_distance_iqr:.1f} IQR. "
        f"Il est conservé comme signal statistique "
        f"à examiner."
    )


# ============================================================
# SINGLE CANDIDATE
# ============================================================


def _build_profile(
    candidate:
        EntityOutlierCandidate,
    *,
    priority_min_signal_count:
        int,
    priority_min_max_distance_iqr:
        float,
    strong_min_max_distance_iqr:
        float,
) -> EntityOutlierProfile:
    evidence = [
        _profile_evidence(
            item
        )

        for item in (
            candidate.evidence
        )
    ]


    evidence.sort(
        key=lambda item:
            item.score,
        reverse=True,
    )


    signal_families = list(
        dict.fromkeys(
            item.family

            for item in (
                evidence
            )
        )
    )


    max_distance_iqr = max(
        item.distance_iqr

        for item in evidence
    )


    dominant_family = (
        _dominant_family(
            evidence
        )
    )


    severity = (
        _severity(
            max_distance_iqr=
                max_distance_iqr,

            signal_count=
                candidate
                .outlier_metric_count,

            priority_min_signal_count=
                priority_min_signal_count,

            priority_min_max_distance_iqr=
                priority_min_max_distance_iqr,

            strong_min_max_distance_iqr=
                strong_min_max_distance_iqr,
        )
    )


    profile_kind = (
        _profile_kind(
            severity=
                severity
        )
    )


    return (
        EntityOutlierProfile(
            entity=
                candidate.entity,

            profile_kind=
                profile_kind,

            severity=
                severity,

            anomaly_score=
                candidate
                .anomaly_score,

            signal_count=
                candidate
                .outlier_metric_count,

            signal_families=
                signal_families,

            dominant_family=
                dominant_family,

            max_distance_iqr=
                max_distance_iqr,

            title=
                _profile_title(
                    profile_kind=
                        profile_kind,

                    dominant_family=
                        dominant_family,
                ),

            rationale=
                _profile_rationale(
                    profile_kind=
                        profile_kind,

                    severity=
                        severity,

                    signal_count=
                        candidate
                        .outlier_metric_count,

                    signal_families=
                        signal_families,

                    dominant_family=
                        dominant_family,

                    max_distance_iqr=
                        max_distance_iqr,
                ),

            evidence=
                evidence,
        )
    )


# ============================================================
# SORTING
# ============================================================


def _profile_sort_key(
    profile:
        EntityOutlierProfile,
) -> tuple[
    int,
    float,
    int,
    float,
]:
    severity_rank = {
        "extreme":
            3,

        "strong":
            2,

        "moderate":
            1,
    }


    return (
        severity_rank[
            profile.severity
        ],

        profile.max_distance_iqr,

        profile.signal_count,

        profile.anomaly_score,
    )


# ============================================================
# PUBLIC API
# ============================================================


def build_entity_outlier_profiles(
    report:
        EntityOutlierReport,
    *,
    priority_min_signal_count:
        int = (
            DEFAULT_PRIORITY_MIN_SIGNAL_COUNT
        ),
    priority_min_max_distance_iqr:
        float = (
            DEFAULT_PRIORITY_MIN_MAX_DISTANCE_IQR
        ),
    strong_min_max_distance_iqr:
        float = (
            DEFAULT_STRONG_MIN_MAX_DISTANCE_IQR
        ),
    top_limit:
        int = (
            DEFAULT_PROFILE_TOP_LIMIT
        ),
) -> EntityOutlierProfileReport:
    """
    Convert raw entity-outlier signals into user-facing
    behavioural profiles.

    This layer does NOT recompute outliers.

    It consumes the deterministic output of
    entity_outlier_engine_v0.1 and adds:

    - behavioural metric families;
    - severity;
    - priority classification;
    - deterministic explanations.

    A raw IQR signal is therefore not automatically presented
    as a globally atypical entity.
    """

    if (
        priority_min_signal_count
        <
        1
    ):
        raise ValueError(
            (
                "priority_min_signal_count "
                "must be greater than zero."
            )
        )


    if (
        priority_min_max_distance_iqr
        <=
        0.0
    ):
        raise ValueError(
            (
                "priority_min_max_distance_iqr "
                "must be greater than zero."
            )
        )


    if (
        strong_min_max_distance_iqr
        <=
        0.0
    ):
        raise ValueError(
            (
                "strong_min_max_distance_iqr "
                "must be greater than zero."
            )
        )


    if (
        strong_min_max_distance_iqr
        >=
        priority_min_max_distance_iqr
    ):
        raise ValueError(
            (
                "strong_min_max_distance_iqr "
                "must be lower than "
                "priority_min_max_distance_iqr."
            )
        )


    if (
        top_limit
        <
        1
    ):
        raise ValueError(
            "top_limit must be greater than zero."
        )


    results: list[
        EntityOutlierProfileViewResult
    ] = []


    for view_result in (
        report.results
    ):
        profiles = [
            _build_profile(
                candidate,

                priority_min_signal_count=
                    priority_min_signal_count,

                priority_min_max_distance_iqr=
                    priority_min_max_distance_iqr,

                strong_min_max_distance_iqr=
                    strong_min_max_distance_iqr,
            )

            for candidate
            in view_result.top_entities
        ]


        profiles.sort(
            key=
                _profile_sort_key,

            reverse=True,
        )


        priority_profiles = [
            profile

            for profile
            in profiles

            if (
                profile.profile_kind
                ==
                "priority_profile"
            )
        ]


        behavioral_signals = [
            profile

            for profile
            in profiles

            if (
                profile.profile_kind
                ==
                "behavioral_signal"
            )
        ]


        classified_entity_count = (
            len(
                profiles
            )
        )


        unclassified_flagged_entity_count = max(
            0,

            (
                view_result
                .flagged_entity_count

                -

                classified_entity_count
            ),
        )


        results.append(
            EntityOutlierProfileViewResult(
                dataset_id=
                    view_result
                    .dataset_id,

                dataset_filename=
                    view_result
                    .dataset_filename,

                entity_column=
                    view_result
                    .entity_column,

                entity_count=
                    view_result
                    .entity_count,

                source_flagged_entity_count=
                    view_result
                    .flagged_entity_count,

                classified_entity_count=
                    classified_entity_count,

                priority_profile_count=
                    len(
                        priority_profiles
                    ),

                behavioral_signal_count=
                    len(
                        behavioral_signals
                    ),

                unclassified_flagged_entity_count=
                    unclassified_flagged_entity_count,

                priority_profiles=
                    priority_profiles[
                        :top_limit
                    ],

                behavioral_signals=
                    behavioral_signals[
                        :top_limit
                    ],
            )
        )


    source_flagged_entity_count = sum(
        result
        .source_flagged_entity_count

        for result
        in results
    )


    classified_entity_count = sum(
        result
        .classified_entity_count

        for result
        in results
    )


    priority_profile_count = sum(
        result
        .priority_profile_count

        for result
        in results
    )


    behavioral_signal_count = sum(
        result
        .behavioral_signal_count

        for result
        in results
    )


    unclassified_flagged_entity_count = sum(
        result
        .unclassified_flagged_entity_count

        for result
        in results
    )


    notes = [
        (
            "Un dépassement IQR n'est pas automatiquement "
            "présenté comme un profil comportemental extrême."
        ),

        (
            "Les profils prioritaires combinent plusieurs "
            "signaux avec au moins une distance IQR "
            "exceptionnellement élevée."
        ),

        (
            "Les familles volume, panier et intensité servent "
            "uniquement à expliquer la nature du signal."
        ),

        (
            "Une priorité analytique ne constitue ni une "
            "preuve de fraude, ni une classification BtoB, "
            "ni une justification de suppression automatique."
        ),
    ]


    return (
        EntityOutlierProfileReport(
            source_view_count=
                len(
                    report.results
                ),

            classified_view_count=
                len(
                    results
                ),

            source_flagged_entity_count=
                source_flagged_entity_count,

            classified_entity_count=
                classified_entity_count,

            priority_profile_count=
                priority_profile_count,

            behavioral_signal_count=
                behavioral_signal_count,

            unclassified_flagged_entity_count=
                unclassified_flagged_entity_count,

            results=
                results,

            notes=
                notes,
        )
    )
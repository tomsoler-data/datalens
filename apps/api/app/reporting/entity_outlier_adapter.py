from __future__ import annotations


import hashlib


from typing import (
    Literal,
)


from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


from app.analysis.entity_outlier_profiles import (
    EntityOutlierProfile,
)

from app.analysis.entity_outlier_requests import (
    EntityOutlierRequestReport,
)


# ============================================================
# VERSION
# ============================================================


ENTITY_OUTLIER_ADAPTER_RULE_VERSION = (
    "entity_outlier_report_adapter_v0.1"
)


# ============================================================
# FRIENDLY LABELS
# ============================================================


METRIC_LABELS = {
    "total_spend":
        "Dépenses totales",

    "purchase_sessions":
        "Sessions d'achat",

    "total_items":
        "Articles achetés",

    "average_basket":
        "Panier moyen",

    "median_basket":
        "Panier médian",

    "average_items_per_basket":
        "Articles moyens par panier",
}


FAMILY_LABELS = {
    "volume":
        "Volume d'activité",

    "basket":
        "Panier",

    "intensity":
        "Intensité d'achat",

    "other":
        "Comportement",
}


# ============================================================
# OUTPUT TYPES
# ============================================================


FindingStatus = Literal[
    "ready",
    "blocked",
]


class EntityOutlierFindingEvidence(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    metric: str

    metric_label: str

    family: str

    family_label: str

    value: float

    direction: str

    distance_iqr: float = Field(
        ge=0.0
    )


class EntityOutlierFindingProfile(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    entity: str

    severity: str

    dominant_family: str

    dominant_family_label: str

    signal_count: int = Field(
        ge=1
    )

    max_distance_iqr: float = Field(
        ge=0.0
    )

    title: str

    explanation: str

    evidence: list[
        EntityOutlierFindingEvidence
    ] = Field(
        default_factory=list
    )


class EntityOutlierFinding(
    BaseModel
):
    """
    User-facing representation of an entity-outlier analysis.

    This model intentionally hides the internal anomaly_score.

    The score remains useful internally for ranking, but it is
    not a probability, percentage or business risk score and
    therefore should not be presented as such to the user.
    """

    model_config = ConfigDict(
        extra="forbid"
    )


    analysis_id: str

    status: FindingStatus

    title: str

    family: Literal[
        "entity_outlier"
    ] = "entity_outlier"

    kind: Literal[
        "customer_entity_outlier_detection"
    ] = (
        "customer_entity_outlier_detection"
    )


    dataset_id: (
        str
        | None
    ) = None

    dataset_filename: (
        str
        | None
    ) = None

    entity_column: (
        str
        | None
    ) = None


    entity_count: int = Field(
        default=0,
        ge=0,
    )

    raw_flagged_entity_count: int = Field(
        default=0,
        ge=0,
    )

    priority_profile_count: int = Field(
        default=0,
        ge=0,
    )

    behavioral_signal_count: int = Field(
        default=0,
        ge=0,
    )


    summary: list[
        str
    ] = Field(
        default_factory=list
    )


    priority_profiles: list[
        EntityOutlierFindingProfile
    ] = Field(
        default_factory=list
    )


    caveats: list[
        str
    ] = Field(
        default_factory=list
    )


    methodology: list[
        str
    ] = Field(
        default_factory=list
    )


    blockers: list[
        str
    ] = Field(
        default_factory=list
    )


    adapter_rule_version: str = (
        ENTITY_OUTLIER_ADAPTER_RULE_VERSION
    )


# ============================================================
# LABEL HELPERS
# ============================================================


def metric_label(
    metric: str,
) -> str:
    return (
        METRIC_LABELS.get(
            metric,
            metric.replace(
                "_",
                " ",
            ).strip().capitalize(),
        )
    )


def family_label(
    family: str,
) -> str:
    return (
        FAMILY_LABELS.get(
            family,
            family.replace(
                "_",
                " ",
            ).strip().capitalize(),
        )
    )


# ============================================================
# STABLE ANALYSIS ID
# ============================================================


def build_entity_outlier_analysis_id(
    report:
        EntityOutlierRequestReport,
) -> str:
    payload = (
        "|".join(
            [
                str(
                    report.intent
                    or
                    ""
                ),

                str(
                    report.dataset_id
                    or
                    ""
                ),

                str(
                    report.entity_column
                    or
                    ""
                ),

                str(
                    report.entity_count
                ),
            ]
        )
    )


    digest = (
        hashlib.sha256(
            payload.encode(
                "utf-8"
            )
        )
        .hexdigest()[
            :16
        ]
    )


    return (
        "entity_outlier:"
        f"{digest}"
    )


# ============================================================
# PROFILE ADAPTER
# ============================================================


def adapt_profile(
    profile:
        EntityOutlierProfile,
) -> EntityOutlierFindingProfile:
    evidence = [
        EntityOutlierFindingEvidence(
            metric=
                item.metric,

            metric_label=
                metric_label(
                    item.metric
                ),

            family=
                item.family,

            family_label=
                family_label(
                    item.family
                ),

            value=
                item.value,

            direction=
                item.direction,

            distance_iqr=
                item.distance_iqr,
        )

        for item
        in profile.evidence
    ]


    return (
        EntityOutlierFindingProfile(
            entity=
                profile.entity,

            severity=
                profile.severity,

            dominant_family=
                profile.dominant_family,

            dominant_family_label=
                family_label(
                    profile
                    .dominant_family
                ),

            signal_count=
                profile.signal_count,

            max_distance_iqr=
                profile
                .max_distance_iqr,

            title=
                profile.title,

            explanation=
                profile.rationale,

            evidence=
                evidence,
        )
    )


# ============================================================
# READY SUMMARY
# ============================================================


def build_ready_summary(
    report:
        EntityOutlierRequestReport,
) -> list[
    str
]:
    summary: list[
        str
    ] = []


    if (
        report.priority_profile_count
        ==
        0
    ):
        summary.append(
            (
                "Aucun profil client extrêmement "
                "atypique n'a été identifié selon "
                "les règles de priorisation actuelles."
            )
        )


    elif (
        report.priority_profile_count
        ==
        1
    ):
        summary.append(
            (
                "1 client présente un profil "
                "d'activité extrêmement atypique "
                "et mérite une revue prioritaire."
            )
        )


    else:
        summary.append(
            (
                f"{report.priority_profile_count} "
                "clients présentent un profil "
                "d'activité extrêmement atypique "
                "et méritent une revue prioritaire."
            )
        )


    if (
        report.raw_flagged_entity_count
        >
        0
    ):
        summary.append(
            (
                f"{report.raw_flagged_entity_count} "
                "client(s) franchissent au moins "
                "une borne statistique IQR."
            )
        )


    if (
        report.behavioral_signal_count
        >
        0
    ):
        summary.append(
            (
                f"{report.behavioral_signal_count} "
                "autre(s) client(s) présentent "
                "des signaux comportementaux "
                "plus localisés."
            )
        )


    summary.append(
        (
            "Les profils prioritaires sont "
            "identifiés au grain client à partir "
            "de métriques comportementales "
            "agrégées."
        )
    )


    return (
        summary
    )


# ============================================================
# METHODOLOGY
# ============================================================


def build_methodology(
    report:
        EntityOutlierRequestReport,
) -> list[
    str
]:
    return [
        (
            "Le grain client est construit par "
            "le moteur de vues analytiques "
            "contrôlées de DataLens."
        ),

        (
            "Chaque métrique comportementale "
            "quantitative est évaluée avec une "
            "règle IQR à 1,5 fois l'intervalle "
            "interquartile."
        ),

        (
            "Les signaux sont ensuite regroupés "
            "par dimensions comportementales : "
            "volume, panier et intensité."
        ),

        (
            "La couche de profilage distingue les "
            "dépassements statistiques isolés des "
            "profils extrêmement atypiques."
        ),

        (
            "Moteurs : "
            f"{report.analytical_view_rule_version}, "
            f"{report.entity_outlier_rule_version}, "
            f"{report.entity_profile_rule_version}."
        ),
    ]


# ============================================================
# CAVEATS
# ============================================================


def build_caveats(
) -> list[
    str
]:
    return [
        (
            "Un profil atypique n'est pas une "
            "preuve de fraude."
        ),

        (
            "Un profil atypique ne permet pas de "
            "conclure automatiquement qu'un client "
            "est professionnel, B2B (BtoB) ou qu'il "
            "appartient à un segment métier particulier."
        ),

        (
            "Aucun client n'est supprimé ou exclu "
            "automatiquement de l'analyse."
        ),

        (
            "Les seuils statistiques doivent être "
            "interprétés avec le contexte métier."
        ),
    ]


# ============================================================
# PUBLIC ADAPTER
# ============================================================


def adapt_entity_outlier_request_to_finding(
    report:
        EntityOutlierRequestReport,
) -> EntityOutlierFinding:
    """
    Convert the internal entity-outlier request contract into
    a stable user-facing finding.

    Important:
    - internal anomaly_score is deliberately omitted;
    - priority profiles remain fully evidenced;
    - secondary signals are counted but are not expanded here;
    - no fraud or B2B inference is added.
    """

    analysis_id = (
        build_entity_outlier_analysis_id(
            report
        )
    )


    if (
        report.status
        !=
        "ready"
    ):
        return (
            EntityOutlierFinding(
                analysis_id=
                    analysis_id,

                status=
                    "blocked",

                title=
                    (
                        "Détection des "
                        "clients atypiques"
                    ),

                dataset_id=
                    report.dataset_id,

                dataset_filename=
                    report.dataset_filename,

                entity_column=
                    report.entity_column,

                entity_count=
                    report.entity_count,

                raw_flagged_entity_count=
                    (
                        report
                        .raw_flagged_entity_count
                    ),

                priority_profile_count=
                    (
                        report
                        .priority_profile_count
                    ),

                behavioral_signal_count=
                    (
                        report
                        .behavioral_signal_count
                    ),

                blockers=
                    list(
                        report.blockers
                    ),

                caveats=
                    build_caveats(),

                methodology=
                    [],
            )
        )


    profiles = [
        adapt_profile(
            profile
        )

        for profile
        in report.priority_profiles
    ]


    return (
        EntityOutlierFinding(
            analysis_id=
                analysis_id,

            status=
                "ready",

            title=
                (
                    "Clients au comportement "
                    "atypique"
                ),

            dataset_id=
                report.dataset_id,

            dataset_filename=
                report.dataset_filename,

            entity_column=
                report.entity_column,

            entity_count=
                report.entity_count,

            raw_flagged_entity_count=
                (
                    report
                    .raw_flagged_entity_count
                ),

            priority_profile_count=
                (
                    report
                    .priority_profile_count
                ),

            behavioral_signal_count=
                (
                    report
                    .behavioral_signal_count
                ),

            summary=
                build_ready_summary(
                    report
                ),

            priority_profiles=
                profiles,

            caveats=
                build_caveats(),

            methodology=
                build_methodology(
                    report
                ),

            blockers=
                [],
        )
    )
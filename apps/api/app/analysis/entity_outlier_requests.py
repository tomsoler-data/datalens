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


from app.analysis.analytical_views import (
    ANALYTICAL_VIEW_RULE_VERSION,
    build_analytical_views,
)

from app.analysis.entity_outliers import (
    ENTITY_OUTLIER_RULE_VERSION,
    detect_entity_outliers,
)

from app.analysis.entity_outlier_profiles import (
    ENTITY_OUTLIER_PROFILE_RULE_VERSION,
    EntityOutlierProfile,
    build_entity_outlier_profiles,
)


# ============================================================
# VERSION
# ============================================================


ENTITY_OUTLIER_REQUEST_RULE_VERSION = (
    "entity_outlier_request_v0.1"
)


ENTITY_OUTLIER_REQUEST_MODEL = (
    "python:entity_outlier_request_v0.1"
)


# ============================================================
# SUPPORTED INTENTS
# ============================================================


EntityOutlierIntent = Literal[
    "customer_entity_outlier_detection",
]


EntityKind = Literal[
    "customer",
]


IntentResolutionStatus = Literal[
    "matched",
    "not_matched",
]


RequestExecutionStatus = Literal[
    "ready",
    "blocked",
    "not_matched",
]


# ============================================================
# INTENT RESOLUTION MODEL
# ============================================================


class EntityOutlierIntentResolution(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    status: IntentResolutionStatus

    objective: str

    normalized_objective: str

    intent: (
        EntityOutlierIntent
        | None
    ) = None

    entity_kind: (
        EntityKind
        | None
    ) = None

    reason: str

    rule_version: str = (
        ENTITY_OUTLIER_REQUEST_RULE_VERSION
    )


# ============================================================
# REQUEST REPORT
# ============================================================


class EntityOutlierRequestReport(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    status: RequestExecutionStatus

    objective: str

    intent: (
        EntityOutlierIntent
        | None
    ) = None

    entity_kind: (
        EntityKind
        | None
    ) = None

    model: str = (
        ENTITY_OUTLIER_REQUEST_MODEL
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


    blockers: list[
        str
    ] = Field(
        default_factory=list
    )


    notes: list[
        str
    ] = Field(
        default_factory=list
    )


    intent_rule_version: str = (
        ENTITY_OUTLIER_REQUEST_RULE_VERSION
    )

    analytical_view_rule_version: str = (
        ANALYTICAL_VIEW_RULE_VERSION
    )

    entity_outlier_rule_version: str = (
        ENTITY_OUTLIER_RULE_VERSION
    )

    entity_profile_rule_version: str = (
        ENTITY_OUTLIER_PROFILE_RULE_VERSION
    )


# ============================================================
# TEXT NORMALIZATION
# ============================================================


def _normalize_text(
    value: str,
) -> str:
    normalized = (
        unicodedata.normalize(
            "NFKD",
            value,
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


    normalized = re.sub(
        r"[^a-z0-9]+",
        " ",
        normalized,
    )


    return (
        re.sub(
            r"\s+",
            " ",
            normalized,
        )
        .strip()
    )


# ============================================================
# INTENT PATTERNS
# ============================================================


CUSTOMER_PATTERN = re.compile(
    (
        r"\b("
        r"client|clients|"
        r"customer|customers|"
        r"acheteur|acheteurs|"
        r"buyer|buyers"
        r")\b"
    )
)


OUTLIER_PATTERN = re.compile(
    (
        r"\b("
        # ----------------------------------------------------
        # FRENCH
        # ----------------------------------------------------

        r"atypique|atypiques|"
        r"anormal|anormaux|anormale|anormales|"
        r"anomalie|anomalies|"
        r"inhabituel|inhabituels|"
        r"inhabituelle|inhabituelles|"
        r"aberrant|aberrants|"
        r"aberrante|aberrantes|"
        r"extreme|extremes|"

        # ----------------------------------------------------
        # ENGLISH
        # ----------------------------------------------------

        r"outlier|outliers|"
        r"anomaly|anomalies|"
        r"anomalous|"
        r"abnormal|"
        r"unusual|"
        r"atypical|"
        r"extreme|extremes"

        r")\b"
    )
)


# ============================================================
# INTENT RESOLVER
# ============================================================


def resolve_entity_outlier_intent(
    objective: str,
) -> EntityOutlierIntentResolution:
    """
    Resolve explicit customer/entity outlier requests.

    v0.1 is deliberately conservative.

    It requires BOTH:

    - an explicit customer/entity target;
    - an explicit anomaly/outlier concept.

    Therefore:

        "Détecte les clients atypiques"
            -> MATCHED

        "Find anomalous customers"
            -> MATCHED

        "Détecte les outliers"
            -> NOT MATCHED

        "Détecte les prix atypiques"
            -> NOT MATCHED

    The generic variable-level outlier resolver remains
    responsible for broad requests until the future
    multi-scope outlier router is introduced.
    """

    raw_objective = str(
        objective
        or
        ""
    ).strip()


    normalized = (
        _normalize_text(
            raw_objective
        )
    )


    if not normalized:
        return (
            EntityOutlierIntentResolution(
                status=
                    "not_matched",

                objective=
                    raw_objective,

                normalized_objective=
                    normalized,

                reason=
                    (
                        "The analytical objective "
                        "is empty."
                    ),
            )
        )


    has_customer_target = bool(
        CUSTOMER_PATTERN.search(
            normalized
        )
    )


    has_outlier_concept = bool(
        OUTLIER_PATTERN.search(
            normalized
        )
    )


    if (
        has_customer_target
        and
        has_outlier_concept
    ):
        return (
            EntityOutlierIntentResolution(
                status=
                    "matched",

                objective=
                    raw_objective,

                normalized_objective=
                    normalized,

                intent=
                    (
                        "customer_entity_"
                        "outlier_detection"
                    ),

                entity_kind=
                    "customer",

                reason=
                    (
                        "The request explicitly "
                        "targets customers and "
                        "an atypical/anomalous "
                        "behaviour concept."
                    ),
            )
        )


    if (
        has_outlier_concept
        and
        not has_customer_target
    ):
        reason = (
            "An outlier concept was detected, "
            "but no customer/entity target was "
            "explicitly requested."
        )


    elif (
        has_customer_target
        and
        not has_outlier_concept
    ):
        reason = (
            "A customer target was detected, "
            "but the request does not explicitly "
            "ask for atypical or anomalous "
            "profiles."
        )


    else:
        reason = (
            "The request does not match the "
            "customer entity-outlier intent."
        )


    return (
        EntityOutlierIntentResolution(
            status=
                "not_matched",

            objective=
                raw_objective,

            normalized_objective=
                normalized,

            reason=
                reason,
        )
    )


# ============================================================
# CUSTOMER VIEW IDENTIFICATION
# ============================================================


def _customer_view_score(
    record: dict[
        str,
        Any,
    ],
) -> int:
    dataframe = (
        record.get(
            "dataframe"
        )
    )


    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        return 0


    preferred_columns = {
        "client_id",
        "total_spend",
        "purchase_sessions",
        "average_basket",
        "median_basket",
        "total_items",
        "average_items_per_basket",
    }


    columns = {
        str(
            column
        )

        for column
        in dataframe.columns
    }


    return len(
        preferred_columns
        &
        columns
    )


def _is_customer_behavior_view(
    record: dict[
        str,
        Any,
    ],
) -> bool:
    dataframe = (
        record.get(
            "dataframe"
        )
    )


    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        return False


    if dataframe.empty:
        return False


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


    entity_column = str(
        provenance.get(
            "entity_column"
        )
        or
        ""
    ).strip()


    columns = {
        str(
            column
        )

        for column
        in dataframe.columns
    }


    if (
        operation
        ==
        "customer_behavior_materialization"
    ):
        return True


    if (
        entity_column
        ==
        "client_id"

        and

        "total_spend"
        in columns
    ):
        return True


    return False


def _select_customer_behavior_view(
    derived_datasets: list[
        dict[
            str,
            Any,
        ]
    ],
) -> (
    dict[
        str,
        Any,
    ]
    | None
):
    candidates = [
        record

        for record
        in derived_datasets

        if (
            _is_customer_behavior_view(
                record
            )
        )
    ]


    if not candidates:
        return None


    candidates.sort(
        key=lambda record: (
            _customer_view_score(
                record
            ),

            len(
                record[
                    "dataframe"
                ]
            ),
        ),
        reverse=True,
    )


    return (
        candidates[
            0
        ]
    )


# ============================================================
# BLOCKED REPORT
# ============================================================


def _blocked_report(
    *,
    objective: str,
    resolution:
        EntityOutlierIntentResolution,
    blocker: str,
    notes: (
        list[
            str
        ]
        | None
    ) = None,
) -> EntityOutlierRequestReport:
    return (
        EntityOutlierRequestReport(
            status=
                "blocked",

            objective=
                objective,

            intent=
                resolution.intent,

            entity_kind=
                resolution.entity_kind,

            blockers=[
                blocker
            ],

            notes=
                list(
                    notes
                    or
                    []
                ),
        )
    )


# ============================================================
# PUBLIC EXECUTION
# ============================================================


def run_entity_outlier_request(
    *,
    objective: str,

    source_dataset_records: list[
        dict[
            str,
            Any,
        ]
    ],

    top_profile_limit: int = 50,
) -> EntityOutlierRequestReport:
    """
    Execute a specific customer entity-outlier request.

    Flow:

        objective
            ↓
        deterministic intent resolution
            ↓
        Analytical View Builder
            ↓
        validated customer-grain view
            ↓
        Entity Outlier Engine
            ↓
        Entity Outlier Profile layer
            ↓
        priority profiles + secondary signals

    No LLM is required for this explicit v0.1 intent.

    No raw transaction row is treated as one customer.
    """

    if (
        top_profile_limit
        <
        1
    ):
        raise ValueError(
            (
                "top_profile_limit must "
                "be greater than zero."
            )
        )


    resolution = (
        resolve_entity_outlier_intent(
            objective
        )
    )


    if (
        resolution.status
        !=
        "matched"
    ):
        return (
            EntityOutlierRequestReport(
                status=
                    "not_matched",

                objective=
                    str(
                        objective
                        or
                        ""
                    ).strip(),

                notes=[
                    resolution.reason
                ],
            )
        )


    if not source_dataset_records:
        return (
            _blocked_report(
                objective=
                    objective,

                resolution=
                    resolution,

                blocker=
                    (
                        "No source dataset "
                        "was provided."
                    ),
            )
        )


    # ========================================================
    # 1. BUILD SAFE ANALYTICAL VIEWS
    # ========================================================

    try:
        view_build = (
            build_analytical_views(
                source_dataset_records
            )
        )


    except Exception as error:
        return (
            _blocked_report(
                objective=
                    objective,

                resolution=
                    resolution,

                blocker=
                    (
                        "The controlled Analytical "
                        "View Builder could not "
                        "materialize the entity "
                        "analysis context."
                    ),

                notes=[
                    (
                        f"{type(error).__name__}: "
                        f"{error}"
                    )
                ],
            )
        )


    # ========================================================
    # 2. SELECT CUSTOMER-GRAIN VIEW
    # ========================================================

    customer_view = (
        _select_customer_behavior_view(
            view_build
            .derived_datasets
        )
    )


    if (
        customer_view
        is None
    ):
        return (
            _blocked_report(
                objective=
                    objective,

                resolution=
                    resolution,

                blocker=
                    (
                        "No validated customer-"
                        "grain behavioural view "
                        "could be materialized "
                        "from the supplied datasets."
                    ),

                notes=[
                    (
                        "DataLens does not silently "
                        "aggregate arbitrary raw rows "
                        "when the customer grain "
                        "cannot be established."
                    )
                ],
            )
        )


    customer_dataframe = (
        customer_view[
            "dataframe"
        ]
    )


    provenance = (
        customer_view.get(
            "provenance"
        )
        or {}
    )


    entity_column = str(
        provenance.get(
            "entity_column"
        )
        or
        "client_id"
    ).strip()


    if (
        entity_column
        not in
        customer_dataframe.columns
    ):
        return (
            _blocked_report(
                objective=
                    objective,

                resolution=
                    resolution,

                blocker=
                    (
                        "The selected customer "
                        "analytical view does not "
                        "contain its declared "
                        "entity column."
                    ),
            )
        )


    # ========================================================
    # 3. RAW ENTITY-OUTLIER DETECTION
    #
    # One row = one customer in this controlled view.
    #
    # Using the entity count as the detector limit guarantees
    # that the profile layer receives every flagged customer,
    # not merely the first N.
    # ========================================================

    detector_limit = max(
        1,
        len(
            customer_dataframe
        ),
    )


    raw_report = (
        detect_entity_outliers(
            datasets=[
                customer_view
            ],

            top_limit=
                detector_limit,
        )
    )


    if (
        not raw_report.results
    ):
        return (
            _blocked_report(
                objective=
                    objective,

                resolution=
                    resolution,

                blocker=
                    (
                        "The customer analytical "
                        "view did not contain enough "
                        "eligible quantitative "
                        "behavioural metrics for "
                        "entity-outlier detection."
                    ),
            )
        )


    raw_customer_result = (
        raw_report.results[
            0
        ]
    )


    # ========================================================
    # 4. PROFILE CLASSIFICATION
    # ========================================================

    profile_report = (
        build_entity_outlier_profiles(
            raw_report,

            top_limit=
                top_profile_limit,
        )
    )


    if (
        not profile_report.results
    ):
        return (
            _blocked_report(
                objective=
                    objective,

                resolution=
                    resolution,

                blocker=
                    (
                        "Entity-outlier signals were "
                        "computed, but no profile "
                        "classification could be "
                        "produced."
                    ),
            )
        )


    profile_result = (
        profile_report.results[
            0
        ]
    )


    # ========================================================
    # 5. USER-FACING STRUCTURED RESULT
    # ========================================================

    notes = [
        (
            "The request was resolved "
            "deterministically as a customer "
            "entity-outlier analysis."
        ),

        (
            "The customer grain was materialized "
            "by the controlled Analytical View "
            "Builder before anomaly detection."
        ),

        (
            f"{profile_result.source_flagged_entity_count} "
            "customer(s) crossed at least one "
            "raw IQR boundary."
        ),

        (
            f"{profile_result.priority_profile_count} "
            "customer(s) were promoted to "
            "extreme priority profiles."
        ),

        (
            f"{profile_result.behavioral_signal_count} "
            "customer(s) remain secondary "
            "behavioural signals."
        ),

        (
            "Priority does not imply fraud, "
            "BtoB status, invalid data or "
            "automatic deletion."
        ),
    ]


    return (
        EntityOutlierRequestReport(
            status=
                "ready",

            objective=
                str(
                    objective
                    or
                    ""
                ).strip(),

            intent=
                resolution.intent,

            entity_kind=
                resolution.entity_kind,

            dataset_id=
                str(
                    customer_view.get(
                        "dataset_id"
                    )
                    or
                    ""
                ),

            dataset_filename=
                str(
                    customer_view.get(
                        "filename"
                    )
                    or
                    ""
                ),

            entity_column=
                raw_customer_result
                .entity_column,

            entity_count=
                raw_customer_result
                .entity_count,

            raw_flagged_entity_count=
                profile_result
                .source_flagged_entity_count,

            priority_profile_count=
                profile_result
                .priority_profile_count,

            behavioral_signal_count=
                profile_result
                .behavioral_signal_count,

            priority_profiles=
                profile_result
                .priority_profiles,

            behavioral_signals=
                profile_result
                .behavioral_signals,

            notes=
                notes,
        )
    )
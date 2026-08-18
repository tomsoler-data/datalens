from __future__ import annotations


import re
import unicodedata

from typing import (
    Any,
)


import numpy as np
import pandas as pd


from app.execution.executor import (
    execute_analysis_candidate,
)

from app.execution.requested_schemas import (
    RequestedAnalysisExecution,
    RequestedAnalysisExecutionReport,
)

from app.execution.schemas import (
    ExecutedAnalysis,
)

from app.execution.structure import (
    detect_observation_structure,
)

from app.planning.schemas import (
    AnalysisCandidate,
    PlannedVariable,
    RequestedAnalysisPlan,
    RequestedAnalysisPlanReport,
)

from app.statistics.executor import (
    calculate_pearson_statistic,
    calculate_spearman_statistic,
)


# ============================================================
# VERSION
# ============================================================

REQUESTED_ANALYSIS_EXECUTOR_RULE_VERSION = (
    "requested_analysis_executor_v0.6"
)


# ============================================================
# SUPPORTED REQUESTS
# ============================================================

SUPPORTED_QUANTITATIVE_REQUESTS = {
    "age_total_amount_association":
        "total_spend",

    "age_frequency_association":
        "purchase_sessions",

    "age_average_basket_association":
        "average_basket",
}


AGE_COLUMN = (
    "age_at_first_purchase"
)


GENDER_COLUMN = (
    "gender"
)


CATEGORY_COLUMN = (
    "category"
)


CUSTOMER_COLUMN = (
    "customer_id"
)


EVENT_TIME_COLUMN = (
    "event_time"
)


MAX_CHART_POINTS = (
    2000
)


DEFAULT_REVENUE_PERIOD = (
    "month"
)


DEFAULT_MOVING_AVERAGE_WINDOW = (
    3
)


DEFAULT_PRODUCT_RANKING_LIMIT = (
    10
)


MAX_LORENZ_CHART_POINTS = (
    1000
)


# ============================================================
# TEXT HELPERS
# ============================================================

def normalize_text(
    value: object,
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
            "ascii",
        )
        .casefold()
    )


    text = re.sub(
        r"[^a-z0-9]+",
        "_",
        text,
    )


    return (
        text.strip(
            "_"
        )
    )


# ============================================================
# DATASET HELPERS
# ============================================================

def dataset_dataframe(
    record: dict[
        str,
        Any,
    ],
) -> (
    pd.DataFrame
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


    return dataframe


def is_customer_entity_column(
    column: (
        str
        | None
    ),
) -> bool:
    if column is None:
        return False


    normalized = (
        normalize_text(
            column
        )
    )


    tokens = set(
        normalized.split(
            "_"
        )
    )


    return bool(
        tokens
        &
        {
            "customer",
            "client",
            "user",
            "buyer",
            "acheteur",
        }
    )


def customer_dataset_hint_score(
    record: dict[
        str,
        Any,
    ],
) -> int:
    values = [
        record.get(
            "dataset_id",
            "",
        ),

        record.get(
            "filename",
            "",
        ),

        record.get(
            "derivation_type",
            "",
        ),
    ]


    provenance = (
        record.get(
            "provenance"
        )
    )


    if isinstance(
        provenance,
        dict,
    ):
        values.extend(
            [
                provenance.get(
                    "grain",
                    "",
                ),

                provenance.get(
                    "operation",
                    "",
                ),
            ]
        )


    text = normalize_text(
        " ".join(
            str(
                value
            )

            for value
            in values
        )
    )


    score = 0


    if bool(
        record.get(
            "is_derived",
            False,
        )
    ):
        score += 20


    if (
        "customer"
        in text
        or
        "client"
        in text
    ):
        score += 40


    if (
        "behavior"
        in text
        or
        "behaviour"
        in text
        or
        "comportement"
        in text
    ):
        score += 20


    return score


# ============================================================
# CUSTOMER-GRAIN DATASET RESOLUTION
# ============================================================

def find_customer_grain_dataset(
    *,
    datasets: list[
        dict[
            str,
            Any,
        ]
    ],

    required_columns: set[
        str
    ],
) -> tuple[
    (
        dict[
            str,
            Any,
        ]
        | None
    ),
    (
        str
        | None
    ),
]:
    """
    Find one defensible customer-grain dataframe.

    Safety rules:

    - all required columns must exist;
    - the observation structure must be
      cross-sectional and unique;
    - a customer-like entity column must be
      identified;
    - the entity column must actually be unique;
    - if several compatible datasets remain,
      DataLens abstains rather than selecting one
      arbitrarily.
    """

    candidates: list[
        tuple[
            int,
            dict[
                str,
                Any,
            ],
        ]
    ] = []


    for record in datasets:
        dataframe = (
            dataset_dataframe(
                record
            )
        )


        if dataframe is None:
            continue


        if not required_columns.issubset(
            set(
                str(
                    column
                )

                for column
                in dataframe.columns
            )
        ):
            continue


        try:
            structure = (
                detect_observation_structure(
                    dataframe
                )
            )


        except Exception:
            continue


        if (
            structure.structure_type
            !=
            "cross_sectional_unique"
        ):
            continue


        if not is_customer_entity_column(
            structure.entity_column
        ):
            continue


        entity_column = (
            structure.entity_column
        )


        if (
            entity_column
            not in dataframe.columns
        ):
            continue


        entity_values = (
            dataframe[
                entity_column
            ]
            .dropna()
        )


        if (
            entity_values.empty
        ):
            continue


        if (
            entity_values
            .duplicated()
            .any()
        ):
            continue


        score = (
            customer_dataset_hint_score(
                record
            )
        )


        candidates.append(
            (
                score,
                record,
            )
        )


    if not candidates:
        return (
            None,
            (
                "Aucun dataset analytique au grain "
                "client ne contient toutes les "
                "variables requises avec une clé "
                "client unique."
            ),
        )


    candidates.sort(
        key=lambda item:
            item[
                0
            ],
        reverse=True,
    )


    best_score = (
        candidates[
            0
        ][
            0
        ]
    )


    best_candidates = [
        record

        for (
            score,
            record,
        )
        in candidates

        if (
            score
            ==
            best_score
        )
    ]


    if (
        len(
            best_candidates
        )
        !=
        1
    ):
        filenames = [
            str(
                record.get(
                    "filename",
                    "unknown",
                )
            )

            for record
            in best_candidates
        ]


        return (
            None,
            (
                "Plusieurs datasets analytiques "
                "compatibles au grain client ont "
                "la même priorité. DataLens refuse "
                "d'en sélectionner un "
                "arbitrairement : "
                + ", ".join(
                    filenames
                )
            ),
        )


    return (
        best_candidates[
            0
        ],
        None,
    )


# ============================================================
# REQUESTED-CONTEXT DATASET RESOLUTION
# ============================================================

def find_requested_context_dataset(
    *,
    datasets: list[
        dict[
            str,
            Any,
        ]
    ],

    required_columns: set[
        str
    ],
) -> tuple[
    (
        dict[
            str,
            Any,
        ]
        | None
    ),
    (
        str
        | None
    ),
]:
    candidates: list[
        dict[
            str,
            Any,
        ]
    ] = []


    for record in datasets:
        if (
            str(
                record.get(
                    "derivation_type",
                    "",
                )
            )
            !=
            "requested_event_context"
        ):
            continue


        if bool(
            record.get(
                "discoverable",
                True,
            )
        ):
            continue


        dataframe = (
            dataset_dataframe(
                record
            )
        )


        if dataframe is None:
            continue


        if not required_columns.issubset(
            {
                str(
                    column
                )

                for column
                in dataframe.columns
            }
        ):
            continue


        candidates.append(
            record
        )


    if not candidates:
        return (
            None,
            (
                "Aucune vue de contexte demandée "
                "ne contient toutes les variables "
                "requises."
            ),
        )


    if (
        len(
            candidates
        )
        >
        1
    ):
        filenames = [
            str(
                record.get(
                    "filename",
                    "unknown",
                )
            )

            for record
            in candidates
        ]


        return (
            None,
            (
                "Plusieurs vues de contexte "
                "demandées sont compatibles. "
                "DataLens refuse d'en sélectionner "
                "une arbitrairement : "
                + ", ".join(
                    filenames
                )
            ),
        )


    return (
        candidates[
            0
        ],
        None,
    )



# ============================================================
# REQUESTED BRIEF DATASET RESOLUTION
# ============================================================

def request_match(
    request: RequestedAnalysisPlan,
    concept: str,
):
    matches = [
        match

        for match
        in request.matched_columns

        if (
            str(
                match.concept
            )
            ==
            concept
        )
    ]


    if (
        len(
            matches
        )
        !=
        1
    ):
        return None


    return matches[
        0
    ]


def source_dataset_by_id(
    *,
    datasets: list[
        dict[
            str,
            Any,
        ]
    ],

    dataset_id: str,
) -> (
    dict[
        str,
        Any,
    ]
    | None
):
    matches = [
        record

        for record
        in datasets

        if (
            not bool(
                record.get(
                    "is_derived",
                    False,
                )
            )
            and
            str(
                record.get(
                    "dataset_id",
                    "",
                )
            )
            ==
            dataset_id
        )
    ]


    if (
        len(
            matches
        )
        !=
        1
    ):
        return None


    return matches[
        0
    ]


def single_required_source_dataset(
    *,
    request: RequestedAnalysisPlan,

    datasets: list[
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
    if (
        len(
            request.required_dataset_ids
        )
        !=
        1
    ):
        return None


    return (
        source_dataset_by_id(
            datasets=
                datasets,

            dataset_id=
                str(
                    request.required_dataset_ids[
                        0
                    ]
                ),
        )
    )


def derived_view_by_provenance(
    *,
    datasets: list[
        dict[
            str,
            Any,
        ]
    ],

    derivation_type: str,

    required_provenance: dict[
        str,
        str,
    ],
) -> tuple[
    (
        dict[
            str,
            Any,
        ]
        | None
    ),
    (
        str
        | None
    ),
]:
    candidates: list[
        dict[
            str,
            Any,
        ]
    ] = []


    for record in datasets:
        if (
            str(
                record.get(
                    "derivation_type",
                    "",
                )
            )
            !=
            derivation_type
        ):
            continue


        provenance = (
            record.get(
                "provenance"
            )
        )


        if not isinstance(
            provenance,
            dict,
        ):
            continue


        compatible = True


        for (
            key,
            expected,
        ) in required_provenance.items():
            if (
                str(
                    provenance.get(
                        key,
                        "",
                    )
                )
                !=
                expected
            ):
                compatible = False

                break


        if compatible:
            candidates.append(
                record
            )


    if not candidates:
        return (
            None,
            (
                "Aucune vue analytique dérivée "
                "ne correspond exactement à la "
                "provenance requise."
            ),
        )


    if (
        len(
            candidates
        )
        !=
        1
    ):
        filenames = [
            str(
                record.get(
                    "filename",
                    "unknown",
                )
            )

            for record
            in candidates
        ]


        return (
            None,
            (
                "Plusieurs vues analytiques "
                "correspondent à la même "
                "provenance. DataLens refuse "
                "d'en sélectionner une "
                "arbitrairement : "
                + ", ".join(
                    filenames
                )
            ),
        )


    return (
        candidates[
            0
        ],
        None,
    )


def build_direct_executed_analysis(
    *,
    request: RequestedAnalysisPlan,

    record: dict[
        str,
        Any,
    ],

    family: str,

    chart_type: str,

    summary: list[
        str
    ],

    metrics: dict[
        str,
        Any,
    ],

    chart_data: list[
        dict[
            str,
            Any,
        ]
    ],

    warnings: list[
        str
    ] | None = None,

    limitations: list[
        str
    ] | None = None,
) -> ExecutedAnalysis:
    return (
        ExecutedAnalysis(
            analysis_id=(
                "requested:"
                f"{request.request_id}:"
                f"{request.kind}"
            ),

            dataset_id=
                str(
                    record.get(
                        "dataset_id",
                        "unknown",
                    )
                ),

            dataset_filename=
                str(
                    record.get(
                        "filename",
                        "unknown",
                    )
                ),

            title=
                request.request_text,

            family=
                family,

            planned_readiness=
                "executable_now",

            execution_status=
                "complete",

            chart_type=
                chart_type,

            summary=
                summary,

            metrics=
                metrics,

            chart_data=
                chart_data,

            warnings=
                warnings
                or [],

            limitations=
                limitations
                or [],

            execution_rule_version=
                "requested_direct_descriptive_v0.1",
        )
    )


def wrap_direct_requested_result(
    *,
    request: RequestedAnalysisPlan,

    record: dict[
        str,
        Any,
    ],

    executed: ExecutedAnalysis,

    analytical_grain: str,

    variables: dict[
        str,
        str,
    ],

    limitations: list[
        str
    ] | None = None,
) -> RequestedAnalysisExecution:
    return (
        RequestedAnalysisExecution(
            request_id=
                request.request_id,

            request_text=
                request.request_text,

            kind=
                request.kind,

            plan_status=
                request.status,

            execution_status=
                executed.execution_status,

            inferential_status=
                "not_applicable",

            source_filename=
                request.source_filename,

            source_locator=
                request.source_locator,

            evidence_quote=
                request.evidence_quote,

            dataset_id=
                executed.dataset_id,

            dataset_filename=
                executed.dataset_filename,

            analytical_grain=
                analytical_grain,

            analysis_mode=
                "exploratory",

            variables=
                variables,

            result=
                executed,

            warnings=[
                *executed.warnings,
            ],

            limitations=[
                *executed.limitations,
                *(
                    limitations
                    or []
                ),
            ],

            executor_rule_version=
                REQUESTED_ANALYSIS_EXECUTOR_RULE_VERSION,
        )
    )


def missing_brief_dataset_result(
    request: RequestedAnalysisPlan,
    *,
    variables: dict[
        str,
        str,
    ],
    analytical_grain: str,
    reason: str,
) -> RequestedAnalysisExecution:
    return (
        RequestedAnalysisExecution(
            request_id=
                request.request_id,

            request_text=
                request.request_text,

            kind=
                request.kind,

            plan_status=
                request.status,

            execution_status=
                "needs_information",

            inferential_status=
                "not_applicable",

            source_filename=
                request.source_filename,

            source_locator=
                request.source_locator,

            evidence_quote=
                request.evidence_quote,

            analytical_grain=
                analytical_grain,

            analysis_mode=
                "exploratory",

            variables=
                variables,

            warnings=[
                reason,
            ],

            limitations=[
                (
                    "DataLens n'exécute pas cette "
                    "demande si le dataset ou la "
                    "vue préparée ne peut pas être "
                    "résolu de manière unique et "
                    "traçable."
                )
            ],

            executor_rule_version=
                REQUESTED_ANALYSIS_EXECUTOR_RULE_VERSION,
        )
    )


# ============================================================
# DESCRIPTIVE CORRELATION FALLBACK
# ============================================================

def calculate_descriptive_correlations(
    *,
    dataframe: pd.DataFrame,

    x_column: str,

    y_column: str,
) -> tuple[
    dict[
        str,
        Any,
    ],
    list[
        str
    ],
]:
    """
    Calculate correlation coefficients only.

    This function deliberately does NOT calculate
    or expose p-values.

    Pearson r and Spearman rho are both returned
    as descriptive measures when the inferential
    decision engine refuses to choose a test.

    No coefficient is promoted to the status of
    a selected inferential test.
    """

    warnings: list[
        str
    ] = []


    if (
        x_column
        not in dataframe.columns
        or
        y_column
        not in dataframe.columns
    ):
        return (
            {},
            [
                (
                    "Les variables requises pour "
                    "la description de l'association "
                    "ne sont pas disponibles."
                )
            ],
        )


    pair_frame = pd.DataFrame(
        {
            x_column:
                pd.to_numeric(
                    dataframe[
                        x_column
                    ],
                    errors=
                        "coerce",
                ),

            y_column:
                pd.to_numeric(
                    dataframe[
                        y_column
                    ],
                    errors=
                        "coerce",
                ),
        }
    )


    pair_frame = (
        pair_frame
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )


    n = int(
        len(
            pair_frame
        )
    )


    if (
        n
        <
        3
    ):
        return (
            {},
            [
                (
                    "Moins de trois paires "
                    "numériques complètes sont "
                    "disponibles pour la description "
                    "de l'association."
                )
            ],
        )


    x_unique = int(
        pair_frame[
            x_column
        ]
        .nunique()
    )


    y_unique = int(
        pair_frame[
            y_column
        ]
        .nunique()
    )


    if (
        x_unique
        <
        2
        or
        y_unique
        <
        2
    ):
        return (
            {},
            [
                (
                    "Au moins une variable ne "
                    "présente pas suffisamment de "
                    "variabilité pour calculer un "
                    "coefficient de corrélation."
                )
            ],
        )


    x_values = (
        pair_frame[
            x_column
        ]
        .to_numpy(
            dtype=float
        )
    )


    y_values = (
        pair_frame[
            y_column
        ]
        .to_numpy(
            dtype=float
        )
    )


    try:
        pearson_r = (
            calculate_pearson_statistic(
                x_values,
                y_values,
            )
        )


    except Exception as error:
        pearson_r = None

        warnings.append(
            (
                "Le coefficient descriptif "
                "de Pearson n'a pas pu être "
                "calculé : "
                f"{type(error).__name__}: "
                f"{error}"
            )
        )


    try:
        spearman_rho = (
            calculate_spearman_statistic(
                x_values,
                y_values,
            )
        )


    except Exception as error:
        spearman_rho = None

        warnings.append(
            (
                "Le coefficient descriptif "
                "de Spearman n'a pas pu être "
                "calculé : "
                f"{type(error).__name__}: "
                f"{error}"
            )
        )


    if (
        pearson_r
        is None
        and
        spearman_rho
        is None
    ):
        return (
            {},
            warnings,
        )


    descriptive_statistics = {
        "n":
            n,

        "x_column":
            x_column,

        "y_column":
            y_column,

        "pearson_r":
            (
                float(
                    pearson_r
                )
                if pearson_r
                is not None
                else None
            ),

        "spearman_rho":
            (
                float(
                    spearman_rho
                )
                if spearman_rho
                is not None
                else None
            ),

        "inference_performed":
            False,

        "p_value":
            None,

        "statistically_significant":
            None,

        "interpretation_scope":
            "descriptive_only",
    }


    return (
        descriptive_statistics,
        warnings,
    )


def build_descriptive_scatter_data(
    *,
    dataframe: pd.DataFrame,

    x_column: str,

    y_column: str,

    max_points: int = MAX_CHART_POINTS,
) -> list[
    dict[
        str,
        float,
    ]
]:
    """
    Build deterministic scatter-plot data for a
    descriptive requested analysis.

    This helper does not calculate or modify any
    statistical result. It only prepares x/y values
    for visualization.

    When more than MAX_CHART_POINTS complete numeric
    pairs are available, evenly spaced row positions
    are selected with numpy.linspace. This mirrors
    the deterministic sampling policy already used
    by the main execution engine.
    """

    if (
        x_column
        not in dataframe.columns
        or
        y_column
        not in dataframe.columns
    ):
        return []


    pair_frame = pd.DataFrame(
        {
            x_column:
                pd.to_numeric(
                    dataframe[
                        x_column
                    ],
                    errors=
                        "coerce",
                ),

            y_column:
                pd.to_numeric(
                    dataframe[
                        y_column
                    ],
                    errors=
                        "coerce",
                ),
        }
    )


    pair_frame = (
        pair_frame
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )


    if pair_frame.empty:
        return []


    chart_source = (
        pair_frame
        .copy()
    )


    if (
        len(
            chart_source
        )
        >
        max_points
    ):
        indexes = np.linspace(
            0,
            len(
                chart_source
            )
            -
            1,
            max_points,
            dtype=int,
        )


        chart_source = (
            chart_source.iloc[
                indexes
            ]
        )


    return [
        {
            "x":
                float(
                    row[
                        x_column
                    ]
                ),

            "y":
                float(
                    row[
                        y_column
                    ]
                ),
        }

        for _, row
        in chart_source.iterrows()
    ]


# ============================================================
# WRAPPER HELPERS
# ============================================================

def not_executed_result(
    request: RequestedAnalysisPlan,
) -> RequestedAnalysisExecution:
    return (
        RequestedAnalysisExecution(
            request_id=
                request.request_id,

            request_text=
                request.request_text,

            kind=
                request.kind,

            plan_status=
                request.status,

            execution_status=
                "not_executed",

            inferential_status=
                "not_applicable",

            source_filename=
                request.source_filename,

            source_locator=
                request.source_locator,

            evidence_quote=
                request.evidence_quote,

            warnings=[
                *request.blockers,
            ],

            limitations=[
                (
                    "La demande n'a pas été "
                    "exécutée car son plan "
                    "analytique n'est pas au "
                    "statut ready."
                )
            ],

            executor_rule_version=
                REQUESTED_ANALYSIS_EXECUTOR_RULE_VERSION,
        )
    )


def unsupported_result(
    request: RequestedAnalysisPlan,
) -> RequestedAnalysisExecution:
    return (
        RequestedAnalysisExecution(
            request_id=
                request.request_id,

            request_text=
                request.request_text,

            kind=
                request.kind,

            plan_status=
                request.status,

            execution_status=
                "not_supported_yet",

            inferential_status=
                "not_applicable",

            source_filename=
                request.source_filename,

            source_locator=
                request.source_locator,

            evidence_quote=
                request.evidence_quote,

            warnings=[
                (
                    "Cette demande est résolue par "
                    "le Request Planner mais le "
                    "Requested Analysis Executor "
                    "v0.6 ne prend pas encore en "
                    "charge cette famille."
                )
            ],

            limitations=[
                (
                    "Aucune analyse approximative "
                    "n'est exécutée à la place de "
                    "la demande originale."
                )
            ],

            executor_rule_version=
                REQUESTED_ANALYSIS_EXECUTOR_RULE_VERSION,
        )
    )


def missing_prepared_dataset_result(
    request: RequestedAnalysisPlan,
    *,
    metric_column: str,
    reason: str,
) -> RequestedAnalysisExecution:
    return (
        RequestedAnalysisExecution(
            request_id=
                request.request_id,

            request_text=
                request.request_text,

            kind=
                request.kind,

            plan_status=
                request.status,

            execution_status=
                "needs_information",

            inferential_status=
                "not_selected",

            source_filename=
                request.source_filename,

            source_locator=
                request.source_locator,

            evidence_quote=
                request.evidence_quote,

            analytical_grain=
                "customer",

            analysis_mode=
                "exploratory",

            variables={
                "x":
                    AGE_COLUMN,

                "y":
                    metric_column,
            },

            warnings=[
                reason,
            ],

            limitations=[
                (
                    "DataLens refuse de reconstruire "
                    "ou d'agréger arbitrairement les "
                    "données lorsque la vue "
                    "analytique attendue ne peut "
                    "pas être identifiée de manière "
                    "unique."
                )
            ],

            executor_rule_version=
                REQUESTED_ANALYSIS_EXECUTOR_RULE_VERSION,
        )
    )


def missing_requested_context_result(
    request: RequestedAnalysisPlan,
    *,
    variables: dict[
        str,
        str,
    ],
    reason: str,
) -> RequestedAnalysisExecution:
    return (
        RequestedAnalysisExecution(
            request_id=
                request.request_id,

            request_text=
                request.request_text,

            kind=
                request.kind,

            plan_status=
                request.status,

            execution_status=
                "needs_information",

            inferential_status=
                "not_selected",

            source_filename=
                request.source_filename,

            source_locator=
                request.source_locator,

            evidence_quote=
                request.evidence_quote,

            analytical_grain=
                "event",

            analysis_mode=
                "exploratory",

            variables=
                variables,

            warnings=[
                reason,
            ],

            limitations=[
                (
                    "DataLens exige une vue "
                    "transactionnelle enrichie et "
                    "traçable avant d'exécuter "
                    "cette demande multi-table."
                )
            ],

            executor_rule_version=
                REQUESTED_ANALYSIS_EXECUTOR_RULE_VERSION,
        )
    )


# ============================================================
# CANDIDATE CREATION
# ============================================================

def build_quantitative_candidate(
    *,
    request: RequestedAnalysisPlan,

    dataset_id: str,

    dataset_filename: str,

    metric_column: str,
) -> AnalysisCandidate:
    return (
        AnalysisCandidate(
            analysis_id=(
                "requested:"
                f"{request.request_id}:"
                f"{request.kind}"
            ),

            dataset_id=
                dataset_id,

            dataset_filename=
                dataset_filename,

            title=
                request.request_text,

            family=
                "quantitative_association",

            priority_score=
                100,

            readiness=
                "executable_now",

            variables=[
                PlannedVariable(
                    column=
                        AGE_COLUMN,

                    role=
                        "x",

                    analysis_kind=
                        "quantitative",
                ),

                PlannedVariable(
                    column=
                        metric_column,

                    role=
                        "y",

                    analysis_kind=
                        "quantitative",
                ),
            ],

            chart_type=
                "scatter",

            statistical_strategy=(
                "correlation_decision_engine"
            ),

            reasons=[
                (
                    "Cette analyse provient d'une "
                    "demande documentaire explicite "
                    "et vérifiée."
                ),
                (
                    "Les deux variables sont "
                    "disponibles dans une vue "
                    "analytique au grain client."
                ),
            ],

            limitations=[
                (
                    "Le document demande d'étudier "
                    "une association mais ne "
                    "pré-spécifie ni une relation "
                    "linéaire, ni une relation "
                    "monotone, ni un test "
                    "statistique particulier."
                )
            ],
        )
    )


def build_gender_category_candidate(
    *,
    request: RequestedAnalysisPlan,

    dataset_id: str,

    dataset_filename: str,
) -> AnalysisCandidate:
    return (
        AnalysisCandidate(
            analysis_id=(
                "requested:"
                f"{request.request_id}:"
                f"{request.kind}"
            ),

            dataset_id=
                dataset_id,

            dataset_filename=
                dataset_filename,

            title=
                request.request_text,

            family=
                "categorical_association",

            priority_score=
                100,

            readiness=
                "executable_now",

            variables=[
                PlannedVariable(
                    column=
                        GENDER_COLUMN,

                    role=
                        "x",

                    analysis_kind=
                        "categorical",
                ),

                PlannedVariable(
                    column=
                        CATEGORY_COLUMN,

                    role=
                        "y",

                    analysis_kind=
                        "categorical",
                ),
            ],

            chart_type=
                "heatmap",

            statistical_strategy=(
                "chi_square_or_fisher_"
                "decision_engine"
            ),

            reasons=[
                (
                    "Cette analyse provient d'une "
                    "demande documentaire explicite "
                    "et vérifiée."
                ),
                (
                    "Genre et catégorie ont été "
                    "alignés sur une vue "
                    "transactionnelle enrichie."
                ),
            ],

            limitations=[
                (
                    "Les achats répétés d'un même "
                    "client restent visibles. Le "
                    "moteur doit donc vérifier "
                    "l'indépendance avant toute "
                    "inférence classique."
                )
            ],
        )
    )


def build_age_category_candidate(
    *,
    request: RequestedAnalysisPlan,

    dataset_id: str,

    dataset_filename: str,
) -> AnalysisCandidate:
    return (
        AnalysisCandidate(
            analysis_id=(
                "requested:"
                f"{request.request_id}:"
                f"{request.kind}"
            ),

            dataset_id=
                dataset_id,

            dataset_filename=
                dataset_filename,

            title=
                request.request_text,

            family=
                "group_comparison",

            priority_score=
                100,

            readiness=
                "executable_now",

            variables=[
                PlannedVariable(
                    column=
                        CATEGORY_COLUMN,

                    role=
                        "group",

                    analysis_kind=
                        "categorical",
                ),

                PlannedVariable(
                    column=
                        AGE_COLUMN,

                    role=
                        "value",

                    analysis_kind=
                        "quantitative",
                ),
            ],

            chart_type=
                "boxplot",

            statistical_strategy=(
                "automatic_group_comparison_engine"
            ),

            reasons=[
                (
                    "Cette analyse provient d'une "
                    "demande documentaire explicite "
                    "et vérifiée."
                ),
                (
                    "L'âge reste quantitatif ; "
                    "aucune tranche d'âge "
                    "arbitraire n'est créée."
                ),
            ],

            limitations=[
                (
                    "Les achats répétés d'un même "
                    "client restent visibles. Les "
                    "comparaisons sont donc "
                    "descriptives tant que le plan "
                    "de dépendance n'autorise pas "
                    "une inférence."
                )
            ],
        )
    )


# ============================================================
# QUANTITATIVE REQUEST EXECUTION
# ============================================================

def execute_quantitative_requested_analysis(
    *,
    request: RequestedAnalysisPlan,

    datasets: list[
        dict[
            str,
            Any,
        ]
    ],

    metric_column: str,
) -> RequestedAnalysisExecution:
    required_columns = {
        AGE_COLUMN,
        metric_column,
    }


    (
        record,
        resolution_error,
    ) = find_customer_grain_dataset(
        datasets=
            datasets,

        required_columns=
            required_columns,
    )


    if record is None:
        return (
            missing_prepared_dataset_result(
                request,

                metric_column=
                    metric_column,

                reason=(
                    resolution_error
                    or
                    (
                        "La vue analytique client "
                        "requise n'a pas pu être "
                        "résolue."
                    )
                ),
            )
        )


    dataframe = (
        dataset_dataframe(
            record
        )
    )


    if dataframe is None:
        return (
            missing_prepared_dataset_result(
                request,

                metric_column=
                    metric_column,

                reason=(
                    "Le dataset analytique résolu "
                    "ne contient pas de DataFrame "
                    "exécutable."
                ),
            )
        )


    dataset_id = str(
        record.get(
            "dataset_id",
            "unknown",
        )
    )


    dataset_filename = str(
        record.get(
            "filename",
            "unknown",
        )
    )


    candidate = (
        build_quantitative_candidate(
            request=
                request,

            dataset_id=
                dataset_id,

            dataset_filename=
                dataset_filename,

            metric_column=
                metric_column,
        )
    )


    executed = (
        execute_analysis_candidate(
            candidate,
            dataframe,
        )
    )


    wrapper_status = (
        executed.execution_status
    )


    inferential_status = (
        "executed"
        if (
            executed.execution_status
            ==
            "complete"
            and
            executed.statistical_result
            is not None
        )
        else "not_selected"
    )


    descriptive_statistics: dict[
        str,
        Any,
    ] = {}


    fallback_warnings: list[
        str
    ] = []


    fallback_limitations: list[
        str
    ] = []


    if (
        executed.execution_status
        ==
        "needs_information"
    ):
        (
            descriptive_statistics,
            fallback_warnings,
        ) = calculate_descriptive_correlations(
            dataframe=
                dataframe,

            x_column=
                AGE_COLUMN,

            y_column=
                metric_column,
        )


        if descriptive_statistics:
            wrapper_status = (
                "descriptive_only"
            )


            descriptive_chart_data = (
                build_descriptive_scatter_data(
                    dataframe=
                        dataframe,

                    x_column=
                        AGE_COLUMN,

                    y_column=
                        metric_column,
                )
            )


            executed = (
                executed.model_copy(
                    update={
                        "chart_data":
                            descriptive_chart_data,
                    }
                )
            )


            fallback_limitations.append(
                (
                    "Les coefficients de Pearson "
                    "et de Spearman sont fournis "
                    "uniquement comme mesures "
                    "descriptives de l'association "
                    "observée."
                )
            )


            fallback_limitations.append(
                (
                    "Aucune p-value n'est calculée "
                    "ou interprétée pour ce fallback "
                    "descriptif."
                )
            )


            fallback_limitations.append(
                (
                    "Aucun des deux coefficients "
                    "descriptifs n'est présenté "
                    "comme le test statistique "
                    "sélectionné."
                )
            )


    return (
        RequestedAnalysisExecution(
            request_id=
                request.request_id,

            request_text=
                request.request_text,

            kind=
                request.kind,

            plan_status=
                request.status,

            execution_status=
                wrapper_status,

            inferential_status=
                inferential_status,

            source_filename=
                request.source_filename,

            source_locator=
                request.source_locator,

            evidence_quote=
                request.evidence_quote,

            dataset_id=
                dataset_id,

            dataset_filename=
                dataset_filename,

            analytical_grain=
                "customer",

            analysis_mode=
                "exploratory",

            variables={
                "x":
                    AGE_COLUMN,

                "y":
                    metric_column,
            },

            descriptive_statistics=
                descriptive_statistics,

            result=
                executed,

            warnings=[
                *executed.warnings,
                *fallback_warnings,
            ],

            limitations=[
                *executed.limitations,
                *fallback_limitations,
            ],

            executor_rule_version=
                REQUESTED_ANALYSIS_EXECUTOR_RULE_VERSION,
        )
    )



# ============================================================
# BRIEF — DETERMINISTIC DESCRIPTIVE EXECUTION
# ============================================================

def execute_revenue_moving_average(
    *,
    request: RequestedAnalysisPlan,

    datasets: list[
        dict[
            str,
            Any,
        ]
    ],
) -> RequestedAnalysisExecution:
    amount_match = (
        request_match(
            request,
            "amount",
        )
    )

    time_match = (
        request_match(
            request,
            "time",
        )
    )


    if (
        amount_match is None
        or
        time_match is None
    ):
        return (
            missing_brief_dataset_result(
                request,

                variables={},

                analytical_grain=
                    DEFAULT_REVENUE_PERIOD,

                reason=(
                    "Les colonnes montant et date "
                    "du plan ne sont pas résolues "
                    "de manière unique."
                ),
            )
        )


    (
        record,
        resolution_error,
    ) = derived_view_by_provenance(
        datasets=
            datasets,

        derivation_type=
            "monthly_additive_measure",

        required_provenance={
            "source_time_column":
                str(
                    time_match.column
                ),

            "source_measure_column":
                str(
                    amount_match.column
                ),
        },
    )


    if record is None:
        return (
            missing_brief_dataset_result(
                request,

                variables={
                    "time":
                        str(
                            time_match.column
                        ),

                    "value":
                        str(
                            amount_match.column
                        ),
                },

                analytical_grain=
                    DEFAULT_REVENUE_PERIOD,

                reason=(
                    resolution_error
                    or
                    (
                        "La vue mensuelle du chiffre "
                        "d'affaires n'a pas pu être "
                        "résolue."
                    )
                ),
            )
        )


    dataframe = (
        dataset_dataframe(
            record
        )
    )


    provenance = (
        record.get(
            "provenance",
            {}
        )
    )


    if (
        dataframe is None
        or
        not isinstance(
            provenance,
            dict,
        )
    ):
        return (
            missing_brief_dataset_result(
                request,

                variables={},

                analytical_grain=
                    DEFAULT_REVENUE_PERIOD,

                reason=(
                    "La vue mensuelle résolue "
                    "n'est pas exécutable."
                ),
            )
        )


    time_column = str(
        provenance.get(
            "target_time_column",
            "month",
        )
    )


    measure_column = str(
        provenance.get(
            "target_measure_column",
            (
                "sum_"
                f"{amount_match.column}"
            ),
        )
    )


    if (
        time_column
        not in dataframe.columns
        or
        measure_column
        not in dataframe.columns
    ):
        return (
            missing_brief_dataset_result(
                request,

                variables={
                    "time":
                        time_column,

                    "value":
                        measure_column,
                },

                analytical_grain=
                    DEFAULT_REVENUE_PERIOD,

                reason=(
                    "La vue mensuelle ne contient "
                    "pas les colonnes matérialisées "
                    "attendues."
                ),
            )
        )


    working = pd.DataFrame(
        {
            "period":
                pd.to_datetime(
                    dataframe[
                        time_column
                    ],
                    errors="coerce",
                ),

            "value":
                pd.to_numeric(
                    dataframe[
                        measure_column
                    ],
                    errors="coerce",
                ),
        }
    ).dropna()


    working = (
        working
        .sort_values(
            "period"
        )
        .reset_index(
            drop=True
        )
    )


    if (
        len(
            working
        )
        <
        2
    ):
        return (
            missing_brief_dataset_result(
                request,

                variables={
                    "time":
                        time_column,

                    "value":
                        measure_column,
                },

                analytical_grain=
                    DEFAULT_REVENUE_PERIOD,

                reason=(
                    "Moins de deux périodes "
                    "complètes sont disponibles."
                ),
            )
        )


    working[
        "moving_average"
    ] = (
        working[
            "value"
        ]
        .rolling(
            window=
                DEFAULT_MOVING_AVERAGE_WINDOW,

            min_periods=1,
        )
        .mean()
    )


    chart_data = [
        {
            "period":
                row[
                    "period"
                ].isoformat(),

            "value":
                float(
                    row[
                        "value"
                    ]
                ),

            "moving_average":
                float(
                    row[
                        "moving_average"
                    ]
                ),
        }

        for _, row
        in working.iterrows()
    ]


    metrics = {
        "time_column":
            time_column,

        "measure_column":
            measure_column,

        "valid_observations":
            int(
                len(
                    working
                )
            ),

        "period_count":
            int(
                len(
                    working
                )
            ),

        "aggregation_period":
            DEFAULT_REVENUE_PERIOD,

        "moving_average_window":
            DEFAULT_MOVING_AVERAGE_WINDOW,

        "moving_average_window_unit":
            "periods",

        "total_revenue":
            float(
                working[
                    "value"
                ]
                .sum()
            ),
    }


    limitations = [
        (
            "Le brief autorise le choix de la "
            "granularité temporelle. En l'absence "
            "de paramètre utilisateur dans "
            "l'interface actuelle, DataLens v0.5 "
            "réutilise la vue mensuelle "
            "déterministe."
        ),
        (
            "La fenêtre de moyenne mobile est "
            "fixée explicitement à 3 périodes "
            "dans cette version et devra devenir "
            "configurable dans l'interface."
        ),
    ]


    executed = (
        build_direct_executed_analysis(
            request=
                request,

            record=
                record,

            family=
                "time_series",

            chart_type=
                "line",

            summary=[
                (
                    "Le chiffre d'affaires a été "
                    "agrégé par mois."
                ),
                (
                    "Une moyenne mobile sur "
                    f"{DEFAULT_MOVING_AVERAGE_WINDOW} "
                    "périodes a été calculée de "
                    "manière déterministe."
                ),
            ],

            metrics=
                metrics,

            chart_data=
                chart_data,

            limitations=
                limitations,
        )
    )


    return (
        wrap_direct_requested_result(
            request=
                request,

            record=
                record,

            executed=
                executed,

            analytical_grain=
                "month",

            variables={
                "time":
                    time_column,

                "value":
                    measure_column,
            },

            limitations=[],
        )
    )


def execute_revenue_by_category(
    *,
    request: RequestedAnalysisPlan,

    datasets: list[
        dict[
            str,
            Any,
        ]
    ],
) -> RequestedAnalysisExecution:
    amount_match = (
        request_match(
            request,
            "amount",
        )
    )

    category_match = (
        request_match(
            request,
            "category",
        )
    )


    if (
        amount_match is None
        or
        category_match is None
    ):
        return (
            missing_brief_dataset_result(
                request,

                variables={},

                analytical_grain=
                    "category",

                reason=(
                    "Les colonnes montant et "
                    "catégorie du plan ne sont pas "
                    "résolues de manière unique."
                ),
            )
        )


    (
        record,
        resolution_error,
    ) = derived_view_by_provenance(
        datasets=
            datasets,

        derivation_type=
            "categorical_additive_measure",

        required_provenance={
            "group_column":
                str(
                    category_match.column
                ),

            "source_measure_column":
                str(
                    amount_match.column
                ),
        },
    )


    if record is None:
        return (
            missing_brief_dataset_result(
                request,

                variables={
                    "group":
                        str(
                            category_match.column
                        ),

                    "value":
                        str(
                            amount_match.column
                        ),
                },

                analytical_grain=
                    "category",

                reason=(
                    resolution_error
                    or
                    (
                        "La vue de chiffre d'affaires "
                        "par catégorie n'a pas pu "
                        "être résolue."
                    )
                ),
            )
        )


    dataframe = (
        dataset_dataframe(
            record
        )
    )

    provenance = (
        record.get(
            "provenance",
            {}
        )
    )


    if (
        dataframe is None
        or
        not isinstance(
            provenance,
            dict,
        )
    ):
        return (
            missing_brief_dataset_result(
                request,

                variables={},

                analytical_grain=
                    "category",

                reason=(
                    "La vue catégorielle résolue "
                    "n'est pas exécutable."
                ),
            )
        )


    group_column = str(
        provenance.get(
            "group_column",
            category_match.column,
        )
    )


    measure_column = str(
        provenance.get(
            "target_measure_column",
            (
                "sum_"
                f"{amount_match.column}"
            ),
        )
    )


    if (
        group_column
        not in dataframe.columns
        or
        measure_column
        not in dataframe.columns
    ):
        return (
            missing_brief_dataset_result(
                request,

                variables={
                    "group":
                        group_column,

                    "value":
                        measure_column,
                },

                analytical_grain=
                    "category",

                reason=(
                    "La vue catégorielle ne "
                    "contient pas les colonnes "
                    "matérialisées attendues."
                ),
            )
        )


    working = pd.DataFrame(
        {
            "category":
                dataframe[
                    group_column
                ],

            "value":
                pd.to_numeric(
                    dataframe[
                        measure_column
                    ],
                    errors="coerce",
                ),
        }
    ).dropna()


    if working.empty:
        return (
            missing_brief_dataset_result(
                request,

                variables={
                    "group":
                        group_column,

                    "value":
                        measure_column,
                },

                analytical_grain=
                    "category",

                reason=(
                    "Aucune catégorie avec un "
                    "chiffre d'affaires valide "
                    "n'est disponible."
                ),
            )
        )


    working = (
        working
        .sort_values(
            "value",
            ascending=False,
        )
    )


    total_revenue = float(
        working[
            "value"
        ]
        .sum()
    )


    chart_data = [
        {
            "category":
                str(
                    row[
                        "category"
                    ]
                ),

            "value":
                float(
                    row[
                        "value"
                    ]
                ),
        }

        for _, row
        in working.iterrows()
    ]


    executed = (
        build_direct_executed_analysis(
            request=
                request,

            record=
                record,

            family=
                "aggregate_breakdown",

            chart_type=
                "bar",

            summary=[
                (
                    f"{len(working)} catégorie(s) "
                    "ont été agrégées selon leur "
                    "chiffre d'affaires."
                )
            ],

            metrics={
                "group_column":
                    group_column,

                "measure_column":
                    measure_column,

                "valid_observations":
                    int(
                        len(
                            working
                        )
                    ),

                "category_count":
                    int(
                        len(
                            working
                        )
                    ),

                "total_revenue":
                    total_revenue,
            },

            chart_data=
                chart_data,
        )
    )


    return (
        wrap_direct_requested_result(
            request=
                request,

            record=
                record,

            executed=
                executed,

            analytical_grain=
                "category",

            variables={
                "group":
                    group_column,

                "value":
                    measure_column,
            },
        )
    )


def execute_customers_by_period(
    *,
    request: RequestedAnalysisPlan,

    datasets: list[
        dict[
            str,
            Any,
        ]
    ],
) -> RequestedAnalysisExecution:
    customer_match = (
        request_match(
            request,
            "customer_id",
        )
    )

    time_match = (
        request_match(
            request,
            "time",
        )
    )


    if (
        customer_match is None
        or
        time_match is None
    ):
        return (
            missing_brief_dataset_result(
                request,

                variables={},

                analytical_grain=
                    "month",

                reason=(
                    "L'identifiant client ou la "
                    "date du plan n'est pas résolu "
                    "de manière unique."
                ),
            )
        )


    if (
        str(
            customer_match.dataset_id
        )
        !=
        str(
            time_match.dataset_id
        )
    ):
        return (
            missing_brief_dataset_result(
                request,

                variables={},

                analytical_grain=
                    "month",

                reason=(
                    "Le client et la date ne sont "
                    "pas résolus dans le même "
                    "dataset transactionnel."
                ),
            )
        )


    record = (
        source_dataset_by_id(
            datasets=
                datasets,

            dataset_id=
                str(
                    customer_match.dataset_id
                ),
        )
    )


    dataframe = (
        dataset_dataframe(
            record
        )
        if record
        is not None
        else None
    )


    if (
        record is None
        or
        dataframe is None
    ):
        return (
            missing_brief_dataset_result(
                request,

                variables={
                    "entity":
                        str(
                            customer_match.column
                        ),

                    "time":
                        str(
                            time_match.column
                        ),
                },

                analytical_grain=
                    "month",

                reason=(
                    "Le dataset transactionnel "
                    "résolu n'est pas disponible."
                ),
            )
        )


    customer_column = str(
        customer_match.column
    )

    time_column = str(
        time_match.column
    )


    parsed_time = pd.to_datetime(
        dataframe[
            time_column
        ],
        errors="coerce",
    )


    working = pd.DataFrame(
        {
            "month":
                parsed_time
                .dt
                .to_period(
                    "M"
                )
                .dt
                .to_timestamp(),

            "customer":
                dataframe[
                    customer_column
                ],
        }
    ).dropna()


    if working.empty:
        return (
            missing_brief_dataset_result(
                request,

                variables={
                    "entity":
                        customer_column,

                    "time":
                        time_column,
                },

                analytical_grain=
                    "month",

                reason=(
                    "Aucune paire date/client "
                    "complète n'est disponible."
                ),
            )
        )


    grouped = (
        working
        .groupby(
            "month",
            dropna=True,
        )[
            "customer"
        ]
        .nunique()
        .sort_index()
    )


    chart_data = [
        {
            "period":
                period.isoformat(),

            "value":
                int(
                    value
                ),
        }

        for (
            period,
            value,
        )
        in grouped.items()
    ]


    executed = (
        build_direct_executed_analysis(
            request=
                request,

            record=
                record,

            family=
                "time_series",

            chart_type=
                "line",

            summary=[
                (
                    "Les clients distincts ont été "
                    "comptés pour chaque mois."
                )
            ],

            metrics={
                "time_column":
                    time_column,

                "entity_column":
                    customer_column,

                "measure_column":
                    "distinct_customers",

                "valid_observations":
                    int(
                        len(
                            working
                        )
                    ),

                "period_count":
                    int(
                        len(
                            grouped
                        )
                    ),

                "distinct_customers_total":
                    int(
                        working[
                            "customer"
                        ]
                        .nunique()
                    ),
            },

            chart_data=
                chart_data,
        )
    )


    return (
        wrap_direct_requested_result(
            request=
                request,

            record=
                record,

            executed=
                executed,

            analytical_grain=
                "month",

            variables={
                "time":
                    time_column,

                "entity":
                    customer_column,
            },
        )
    )


def execute_transaction_count(
    *,
    request: RequestedAnalysisPlan,

    datasets: list[
        dict[
            str,
            Any,
        ]
    ],
) -> RequestedAnalysisExecution:
    record = (
        single_required_source_dataset(
            request=
                request,

            datasets=
                datasets,
        )
    )


    dataframe = (
        dataset_dataframe(
            record
        )
        if record
        is not None
        else None
    )


    if (
        record is None
        or
        dataframe is None
    ):
        return (
            missing_brief_dataset_result(
                request,

                variables={},

                analytical_grain=
                    "transaction",

                reason=(
                    "Le dataset transactionnel "
                    "unique du plan n'a pas pu "
                    "être résolu."
                ),
            )
        )


    transaction_count = int(
        len(
            dataframe
        )
    )


    executed = (
        build_direct_executed_analysis(
            request=
                request,

            record=
                record,

            family=
                "descriptive_metric",

            chart_type=
                "metric",

            summary=[
                (
                    f"{transaction_count} événement(s) "
                    "transactionnel(s) sont présents "
                    "dans le dataset préparé."
                )
            ],

            metrics={
                "valid_observations":
                    transaction_count,

                "transaction_count":
                    transaction_count,

                "count_semantics":
                    (
                        "one prepared transaction "
                        "row equals one counted "
                        "transactional event"
                    ),
            },

            chart_data=[],
        )
    )


    return (
        wrap_direct_requested_result(
            request=
                request,

            record=
                record,

            executed=
                executed,

            analytical_grain=
                "transaction",

            variables={},
        )
    )


def execute_products_sold_count(
    *,
    request: RequestedAnalysisPlan,

    datasets: list[
        dict[
            str,
            Any,
        ]
    ],
) -> RequestedAnalysisExecution:
    product_match = (
        request_match(
            request,
            "product_id",
        )
    )


    if product_match is None:
        return (
            missing_brief_dataset_result(
                request,

                variables={},

                analytical_grain=
                    "transaction",

                reason=(
                    "L'identifiant produit du plan "
                    "n'est pas résolu de manière "
                    "unique."
                ),
            )
        )


    record = (
        source_dataset_by_id(
            datasets=
                datasets,

            dataset_id=
                str(
                    product_match.dataset_id
                ),
        )
    )


    dataframe = (
        dataset_dataframe(
            record
        )
        if record
        is not None
        else None
    )


    product_column = str(
        product_match.column
    )


    if (
        record is None
        or
        dataframe is None
        or
        product_column
        not in dataframe.columns
    ):
        return (
            missing_brief_dataset_result(
                request,

                variables={
                    "product":
                        product_column,
                },

                analytical_grain=
                    "transaction",

                reason=(
                    "Le dataset produit/transaction "
                    "résolu n'est pas exécutable."
                ),
            )
        )


    product_events = (
        dataframe[
            product_column
        ]
        .dropna()
    )


    sold_count = int(
        len(
            product_events
        )
    )


    distinct_products = int(
        product_events
        .nunique()
    )


    executed = (
        build_direct_executed_analysis(
            request=
                request,

            record=
                record,

            family=
                "descriptive_metric",

            chart_type=
                "metric",

            summary=[
                (
                    f"{sold_count} occurrence(s) "
                    "produit sont observées dans "
                    "les événements transactionnels."
                ),
                (
                    f"Elles concernent "
                    f"{distinct_products} référence(s) "
                    "produit distincte(s)."
                ),
            ],

            metrics={
                "product_column":
                    product_column,

                "valid_observations":
                    sold_count,

                "products_sold_count":
                    sold_count,

                "distinct_products_sold":
                    distinct_products,

                "count_semantics":
                    (
                        "count of non-null product "
                        "occurrences at transaction "
                        "event grain"
                    ),
            },

            chart_data=[],

            limitations=[
                (
                    "En l'absence d'une colonne "
                    "quantité explicite, une "
                    "occurrence produit dans un "
                    "événement transactionnel est "
                    "comptée comme une unité "
                    "observée."
                )
            ],
        )
    )


    return (
        wrap_direct_requested_result(
            request=
                request,

            record=
                record,

            executed=
                executed,

            analytical_grain=
                "transaction",

            variables={
                "product":
                    product_column,
            },
        )
    )



def find_product_revenue_dataset(
    *,
    request: RequestedAnalysisPlan,

    datasets: list[
        dict[
            str,
            Any,
        ]
    ],
) -> tuple[
    (
        dict[
            str,
            Any,
        ]
        | None
    ),
    (
        str
        | None
    ),
]:
    product_match = (
        request_match(
            request,
            "product_id",
        )
    )

    amount_match = (
        request_match(
            request,
            "amount",
        )
    )


    if (
        product_match is None
        or
        amount_match is None
    ):
        return (
            None,
            (
                "L'identifiant produit ou la "
                "mesure monétaire du plan n'est "
                "pas résolu de manière unique."
            ),
        )


    return (
        derived_view_by_provenance(
            datasets=
                datasets,

            derivation_type=
                "entity_additive_measure",

            required_provenance={
                "entity_column":
                    str(
                        product_match.column
                    ),

                "source_measure_column":
                    str(
                        amount_match.column
                    ),
            },
        )
    )


def execute_product_ranking(
    *,
    request: RequestedAnalysisPlan,

    datasets: list[
        dict[
            str,
            Any,
        ]
    ],

    ascending: bool,
) -> RequestedAnalysisExecution:
    product_match = (
        request_match(
            request,
            "product_id",
        )
    )

    amount_match = (
        request_match(
            request,
            "amount",
        )
    )


    if (
        product_match is None
        or
        amount_match is None
    ):
        return (
            missing_brief_dataset_result(
                request,

                variables={},

                analytical_grain=
                    "product",

                reason=(
                    "L'identifiant produit ou la "
                    "mesure monétaire n'est pas "
                    "résolu de manière unique."
                ),
            )
        )


    (
        record,
        resolution_error,
    ) = find_product_revenue_dataset(
        request=
            request,

        datasets=
            datasets,
    )


    if record is None:
        return (
            missing_brief_dataset_result(
                request,

                variables={
                    "product":
                        str(
                            product_match.column
                        ),

                    "value":
                        str(
                            amount_match.column
                        ),
                },

                analytical_grain=
                    "product",

                reason=(
                    resolution_error
                    or
                    (
                        "La vue produit/chiffre "
                        "d'affaires n'a pas pu "
                        "être résolue."
                    )
                ),
            )
        )


    dataframe = (
        dataset_dataframe(
            record
        )
    )

    provenance = (
        record.get(
            "provenance",
            {}
        )
    )


    if (
        dataframe is None
        or
        not isinstance(
            provenance,
            dict,
        )
    ):
        return (
            missing_brief_dataset_result(
                request,

                variables={},

                analytical_grain=
                    "product",

                reason=(
                    "La vue produit résolue "
                    "n'est pas exécutable."
                ),
            )
        )


    product_column = str(
        provenance.get(
            "entity_column",
            product_match.column,
        )
    )

    revenue_column = str(
        provenance.get(
            "target_measure_column",
            (
                "sum_"
                f"{amount_match.column}"
            ),
        )
    )


    if (
        product_column
        not in dataframe.columns
        or
        revenue_column
        not in dataframe.columns
    ):
        return (
            missing_brief_dataset_result(
                request,

                variables={
                    "product":
                        product_column,

                    "value":
                        revenue_column,
                },

                analytical_grain=
                    "product",

                reason=(
                    "La vue produit ne contient "
                    "pas les colonnes matérialisées "
                    "attendues."
                ),
            )
        )


    working = pd.DataFrame(
        {
            "product":
                dataframe[
                    product_column
                ],

            "revenue":
                pd.to_numeric(
                    dataframe[
                        revenue_column
                    ],
                    errors="coerce",
                ),
        }
    )


    working = (
        working
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )


    if working.empty:
        return (
            missing_brief_dataset_result(
                request,

                variables={
                    "product":
                        product_column,

                    "value":
                        revenue_column,
                },

                analytical_grain=
                    "product",

                reason=(
                    "Aucune référence produit avec "
                    "un chiffre d'affaires valide "
                    "n'est disponible."
                ),
            )
        )


    working = (
        working
        .sort_values(
            "revenue",
            ascending=
                ascending,

            kind=
                "mergesort",
        )
        .reset_index(
            drop=True
        )
    )


    ranking_limit = min(
        DEFAULT_PRODUCT_RANKING_LIMIT,
        len(
            working
        ),
    )


    ranking = (
        working
        .head(
            ranking_limit
        )
        .copy()
    )


    chart_data = [
        {
            "category":
                str(
                    row[
                        "product"
                    ]
                ),

            "value":
                float(
                    row[
                        "revenue"
                    ]
                ),

            "rank":
                int(
                    index +
                    1
                ),
        }

        for (
            index,
            (
                _,
                row,
            ),
        )
        in enumerate(
            ranking.iterrows()
        )
    ]


    ranking_direction = (
        "croissant"
        if ascending
        else "décroissant"
    )


    ranking_label = (
        "flop"
        if ascending
        else "top"
    )


    executed = (
        build_direct_executed_analysis(
            request=
                request,

            record=
                record,

            family=
                "ranking",

            chart_type=
                "bar",

            summary=[
                (
                    f"Les {ranking_limit} référence(s) "
                    f"du {ranking_label} ont été "
                    "classées selon le chiffre "
                    "d'affaires agrégé."
                ),
                (
                    "Le classement est effectué "
                    f"par ordre {ranking_direction}."
                ),
            ],

            metrics={
                "product_column":
                    product_column,

                "measure_column":
                    revenue_column,

                "ranking_direction":
                    ranking_direction,

                "ranking_limit":
                    ranking_limit,

                "ranked_product_count":
                    int(
                        len(
                            working
                        )
                    ),

                "valid_observations":
                    int(
                        len(
                            working
                        )
                    ),
            },

            chart_data=
                chart_data,

            limitations=[
                (
                    "Le brief définit explicitement "
                    "le classement par chiffre "
                    "d'affaires, mais ne précise pas "
                    "le nombre de références à "
                    "afficher."
                ),
                (
                    "DataLens v0.6 affiche les "
                    f"{DEFAULT_PRODUCT_RANKING_LIMIT} "
                    "premières références par "
                    "défaut. Cette limite devra "
                    "devenir configurable dans "
                    "l'interface."
                ),
            ],
        )
    )


    return (
        wrap_direct_requested_result(
            request=
                request,

            record=
                record,

            executed=
                executed,

            analytical_grain=
                "product",

            variables={
                "product":
                    product_column,

                "value":
                    revenue_column,
            },
        )
    )


def execute_product_category_distribution(
    *,
    request: RequestedAnalysisPlan,

    datasets: list[
        dict[
            str,
            Any,
        ]
    ],
) -> RequestedAnalysisExecution:
    product_match = (
        request_match(
            request,
            "product_id",
        )
    )

    category_match = (
        request_match(
            request,
            "category",
        )
    )


    if (
        product_match is None
        or
        category_match is None
    ):
        return (
            missing_brief_dataset_result(
                request,

                variables={},

                analytical_grain=
                    "product",

                reason=(
                    "L'identifiant produit ou la "
                    "catégorie n'est pas résolu "
                    "de manière unique."
                ),
            )
        )


    product_candidates: list[
        dict[
            str,
            Any,
        ]
    ] = []


    for record in datasets:
        if (
            str(
                record.get(
                    "derivation_type",
                    "",
                )
            )
            !=
            "entity_additive_measure"
        ):
            continue


        provenance = (
            record.get(
                "provenance"
            )
        )


        if not isinstance(
            provenance,
            dict,
        ):
            continue


        if (
            str(
                provenance.get(
                    "entity_column",
                    "",
                )
            )
            !=
            str(
                product_match.column
            )
        ):
            continue


        dataframe = (
            dataset_dataframe(
                record
            )
        )


        if (
            dataframe is None
            or
            str(
                category_match.column
            )
            not in dataframe.columns
        ):
            continue


        product_candidates.append(
            record
        )


    if (
        len(
            product_candidates
        )
        !=
        1
    ):
        reason = (
            "Aucune vue produit unique avec la "
            "catégorie demandée n'a été résolue."
        )


        if (
            len(
                product_candidates
            )
            >
            1
        ):
            reason = (
                "Plusieurs vues produit avec la "
                "catégorie demandée sont "
                "compatibles. DataLens refuse "
                "d'en sélectionner une "
                "arbitrairement."
            )


        return (
            missing_brief_dataset_result(
                request,

                variables={
                    "product":
                        str(
                            product_match.column
                        ),

                    "group":
                        str(
                            category_match.column
                        ),
                },

                analytical_grain=
                    "product",

                reason=
                    reason,
            )
        )


    record = (
        product_candidates[
            0
        ]
    )


    dataframe = (
        dataset_dataframe(
            record
        )
    )


    if dataframe is None:
        return (
            missing_brief_dataset_result(
                request,

                variables={},

                analytical_grain=
                    "product",

                reason=(
                    "La vue produit résolue "
                    "n'est pas exécutable."
                ),
            )
        )


    product_column = str(
        product_match.column
    )

    category_column = str(
        category_match.column
    )


    if (
        product_column
        not in dataframe.columns
        or
        category_column
        not in dataframe.columns
    ):
        return (
            missing_brief_dataset_result(
                request,

                variables={
                    "product":
                        product_column,

                    "group":
                        category_column,
                },

                analytical_grain=
                    "product",

                reason=(
                    "La vue produit ne contient "
                    "pas les colonnes attendues."
                ),
            )
        )


    working = (
        dataframe[
            [
                product_column,
                category_column,
            ]
        ]
        .dropna()
        .drop_duplicates(
            subset=[
                product_column
            ]
        )
    )


    if working.empty:
        return (
            missing_brief_dataset_result(
                request,

                variables={
                    "product":
                        product_column,

                    "group":
                        category_column,
                },

                analytical_grain=
                    "product",

                reason=(
                    "Aucune référence produit avec "
                    "une catégorie valide n'est "
                    "disponible."
                ),
            )
        )


    grouped = (
        working
        .groupby(
            category_column,
            dropna=True,
        )[
            product_column
        ]
        .nunique()
        .sort_values(
            ascending=False
        )
    )


    total_references = int(
        working[
            product_column
        ]
        .nunique()
    )


    chart_data = [
        {
            "category":
                str(
                    category
                ),

            "value":
                int(
                    count
                ),

            "share":
                float(
                    count /
                    total_references
                )
                if total_references
                else 0.0,
        }

        for (
            category,
            count,
        )
        in grouped.items()
    ]


    executed = (
        build_direct_executed_analysis(
            request=
                request,

            record=
                record,

            family=
                "categorical_breakdown",

            chart_type=
                "bar",

            summary=[
                (
                    f"{total_references} référence(s) "
                    "distincte(s) observée(s) ont "
                    "été réparties entre "
                    f"{len(grouped)} catégorie(s)."
                )
            ],

            metrics={
                "product_column":
                    product_column,

                "group_column":
                    category_column,

                "reference_count":
                    total_references,

                "category_count":
                    int(
                        len(
                            grouped
                        )
                    ),

                "valid_observations":
                    total_references,
            },

            chart_data=
                chart_data,

            limitations=[
                (
                    "La répartition porte sur les "
                    "références observées dans la "
                    "vue produit construite depuis "
                    "les transactions et le "
                    "référentiel produit."
                )
            ],
        )
    )


    return (
        wrap_direct_requested_result(
            request=
                request,

            record=
                record,

            executed=
                executed,

            analytical_grain=
                "product",

            variables={
                "product":
                    product_column,

                "group":
                    category_column,
            },
        )
    )


def execute_lorenz_curve(
    *,
    request: RequestedAnalysisPlan,

    datasets: list[
        dict[
            str,
            Any,
        ]
    ],
) -> RequestedAnalysisExecution:
    customer_match = (
        request_match(
            request,
            "customer_id",
        )
    )

    amount_match = (
        request_match(
            request,
            "amount",
        )
    )


    if (
        customer_match is None
        or
        amount_match is None
    ):
        return (
            missing_brief_dataset_result(
                request,

                variables={},

                analytical_grain=
                    "customer",

                reason=(
                    "L'identifiant client ou la "
                    "mesure monétaire du plan "
                    "n'est pas résolu de manière "
                    "unique."
                ),
            )
        )


    (
        record,
        resolution_error,
    ) = find_customer_grain_dataset(
        datasets=
            datasets,

        required_columns={
            "total_spend",
        },
    )


    if record is None:
        return (
            missing_brief_dataset_result(
                request,

                variables={
                    "entity":
                        str(
                            customer_match.column
                        ),

                    "value":
                        "total_spend",
                },

                analytical_grain=
                    "customer",

                reason=(
                    resolution_error
                    or
                    (
                        "La vue client contenant "
                        "le chiffre d'affaires total "
                        "n'a pas pu être résolue."
                    )
                ),
            )
        )


    dataframe = (
        dataset_dataframe(
            record
        )
    )


    if (
        dataframe is None
        or
        "total_spend"
        not in dataframe.columns
    ):
        return (
            missing_brief_dataset_result(
                request,

                variables={
                    "value":
                        "total_spend",
                },

                analytical_grain=
                    "customer",

                reason=(
                    "La vue client résolue ne "
                    "contient pas total_spend."
                ),
            )
        )


    total_spend = (
        pd.to_numeric(
            dataframe[
                "total_spend"
            ],
            errors="coerce",
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )


    negative_count = int(
        (
            total_spend
            <
            0
        )
        .sum()
    )


    non_negative = (
        total_spend[
            total_spend
            >=
            0
        ]
        .astype(
            float
        )
        .sort_values(
            kind=
                "mergesort"
        )
        .reset_index(
            drop=True
        )
    )


    if (
        len(
            non_negative
        )
        <
        2
        or
        float(
            non_negative.sum()
        )
        <=
        0
    ):
        return (
            missing_brief_dataset_result(
                request,

                variables={
                    "value":
                        "total_spend",
                },

                analytical_grain=
                    "customer",

                reason=(
                    "La courbe de Lorenz nécessite "
                    "au moins deux valeurs client "
                    "non négatives et une somme "
                    "strictement positive."
                ),
            )
        )


    values = (
        non_negative
        .to_numpy(
            dtype=float
        )
    )


    customer_count = int(
        len(
            values
        )
    )


    total_value = float(
        values.sum()
    )


    cumulative_revenue = (
        np.cumsum(
            values
        )
        /
        total_value
    )


    cumulative_customers = (
        np.arange(
            1,
            customer_count +
                1,
            dtype=float,
        )
        /
        customer_count
    )


    lorenz_x = np.concatenate(
        [
            np.array(
                [
                    0.0
                ],
                dtype=float,
            ),
            cumulative_customers,
        ]
    )


    lorenz_y = np.concatenate(
        [
            np.array(
                [
                    0.0
                ],
                dtype=float,
            ),
            cumulative_revenue,
        ]
    )


    weighted_rank_sum = float(
        np.sum(
            np.arange(
                1,
                customer_count +
                    1,
                dtype=float,
            )
            *
            values
        )
    )


    gini = (
        (
            2.0
            *
            weighted_rank_sum
        )
        /
        (
            customer_count
            *
            total_value
        )
        -
        (
            customer_count +
            1
        )
        /
        customer_count
    )


    gini = float(
        max(
            0.0,
            min(
                1.0,
                gini,
            ),
        )
    )


    if (
        len(
            lorenz_x
        )
        >
        MAX_LORENZ_CHART_POINTS
    ):
        indexes = np.linspace(
            0,
            len(
                lorenz_x
            )
            -
            1,
            MAX_LORENZ_CHART_POINTS,
            dtype=int,
        )


        indexes = np.unique(
            indexes
        )


        chart_x = (
            lorenz_x[
                indexes
            ]
        )

        chart_y = (
            lorenz_y[
                indexes
            ]
        )


    else:
        chart_x = (
            lorenz_x
        )

        chart_y = (
            lorenz_y
        )


    chart_data = [
        {
            "population_share":
                float(
                    population_share
                ),

            "revenue_share":
                float(
                    revenue_share
                ),

            "equality_share":
                float(
                    population_share
                ),
        }

        for (
            population_share,
            revenue_share,
        )
        in zip(
            chart_x,
            chart_y,
        )
    ]


    warnings: list[
        str
    ] = []


    if (
        negative_count
        >
        0
    ):
        warnings.append(
            (
                f"{negative_count} valeur(s) client "
                "négative(s) ont été exclues car "
                "la courbe de Lorenz standard "
                "utilisée ici suppose une mesure "
                "non négative."
            )
        )


    executed = (
        build_direct_executed_analysis(
            request=
                request,

            record=
                record,

            family=
                "inequality",

            chart_type=
                "lorenz",

            summary=[
                (
                    f"La courbe de Lorenz a été "
                    f"construite sur "
                    f"{customer_count} client(s)."
                ),
                (
                    "Les clients sont triés par "
                    "chiffre d'affaires croissant "
                    "avant le calcul des parts "
                    "cumulées."
                ),
            ],

            metrics={
                "entity_column":
                    str(
                        customer_match.column
                    ),

                "measure_column":
                    "total_spend",

                "customer_count":
                    customer_count,

                "valid_observations":
                    customer_count,

                "total_revenue":
                    total_value,

                "gini_coefficient":
                    gini,

                "negative_value_count":
                    negative_count,

                "chart_point_count":
                    int(
                        len(
                            chart_data
                        )
                    ),
            },

            chart_data=
                chart_data,

            warnings=
                warnings,

            limitations=[
                (
                    "Le coefficient de Gini est "
                    "calculé comme résumé descriptif "
                    "de concentration sur le même "
                    "chiffre d'affaires client."
                ),
                (
                    "La visualisation peut être "
                    "sous-échantillonnée de manière "
                    "déterministe, mais le Gini et "
                    "les cumuls sont calculés sur "
                    "toutes les valeurs valides."
                ),
            ],
        )
    )


    return (
        wrap_direct_requested_result(
            request=
                request,

            record=
                record,

            executed=
                executed,

            analytical_grain=
                "customer",

            variables={
                "entity":
                    str(
                        customer_match.column
                    ),

                "value":
                    "total_spend",
            },
        )
    )


def execute_brief_requested_analysis(
    *,
    request: RequestedAnalysisPlan,

    datasets: list[
        dict[
            str,
            Any,
        ]
    ],
) -> RequestedAnalysisExecution:
    if (
        request.kind
        ==
        "revenue_moving_average"
    ):
        return (
            execute_revenue_moving_average(
                request=
                    request,

                datasets=
                    datasets,
            )
        )


    if (
        request.kind
        ==
        "revenue_by_category"
    ):
        return (
            execute_revenue_by_category(
                request=
                    request,

                datasets=
                    datasets,
            )
        )


    if (
        request.kind
        ==
        "customers_by_period"
    ):
        return (
            execute_customers_by_period(
                request=
                    request,

                datasets=
                    datasets,
            )
        )


    if (
        request.kind
        ==
        "transaction_count"
    ):
        return (
            execute_transaction_count(
                request=
                    request,

                datasets=
                    datasets,
            )
        )


    if (
        request.kind
        ==
        "products_sold_count"
    ):
        return (
            execute_products_sold_count(
                request=
                    request,

                datasets=
                    datasets,
            )
        )


    if (
        request.kind
        ==
        "top_products"
    ):
        return (
            execute_product_ranking(
                request=
                    request,

                datasets=
                    datasets,

                ascending=
                    False,
            )
        )


    if (
        request.kind
        ==
        "flop_products"
    ):
        return (
            execute_product_ranking(
                request=
                    request,

                datasets=
                    datasets,

                ascending=
                    True,
            )
        )


    if (
        request.kind
        ==
        "product_category_distribution"
    ):
        return (
            execute_product_category_distribution(
                request=
                    request,

                datasets=
                    datasets,
            )
        )


    if (
        request.kind
        ==
        "lorenz_curve"
    ):
        return (
            execute_lorenz_curve(
                request=
                    request,

                datasets=
                    datasets,
            )
        )


    return (
        unsupported_result(
            request
        )
    )


# ============================================================
# DOCUMENTARY CONTEXT REQUEST EXECUTION
# ============================================================

def execute_context_requested_analysis(
    *,
    request: RequestedAnalysisPlan,

    datasets: list[
        dict[
            str,
            Any,
        ]
    ],
) -> RequestedAnalysisExecution:
    if (
        request.kind
        ==
        "gender_category_association"
    ):
        required_columns = {
            CUSTOMER_COLUMN,
            EVENT_TIME_COLUMN,
            GENDER_COLUMN,
            CATEGORY_COLUMN,
        }

        variables = {
            "x":
                GENDER_COLUMN,

            "y":
                CATEGORY_COLUMN,
        }


    elif (
        request.kind
        ==
        "age_category_association"
    ):
        required_columns = {
            CUSTOMER_COLUMN,
            EVENT_TIME_COLUMN,
            AGE_COLUMN,
            CATEGORY_COLUMN,
        }

        variables = {
            "group":
                CATEGORY_COLUMN,

            "value":
                AGE_COLUMN,
        }


    else:
        return (
            unsupported_result(
                request
            )
        )


    (
        record,
        resolution_error,
    ) = find_requested_context_dataset(
        datasets=
            datasets,

        required_columns=
            required_columns,
    )


    if record is None:
        return (
            missing_requested_context_result(
                request,

                variables=
                    variables,

                reason=(
                    resolution_error
                    or
                    (
                        "La vue de contexte "
                        "demandée n'a pas pu être "
                        "résolue."
                    )
                ),
            )
        )


    dataframe = (
        dataset_dataframe(
            record
        )
    )


    if dataframe is None:
        return (
            missing_requested_context_result(
                request,

                variables=
                    variables,

                reason=(
                    "La vue de contexte résolue "
                    "ne contient pas de DataFrame "
                    "exécutable."
                ),
            )
        )


    dataset_id = str(
        record.get(
            "dataset_id",
            "unknown",
        )
    )


    dataset_filename = str(
        record.get(
            "filename",
            "unknown",
        )
    )


    if (
        request.kind
        ==
        "gender_category_association"
    ):
        candidate = (
            build_gender_category_candidate(
                request=
                    request,

                dataset_id=
                    dataset_id,

                dataset_filename=
                    dataset_filename,
            )
        )


    else:
        candidate = (
            build_age_category_candidate(
                request=
                    request,

                dataset_id=
                    dataset_id,

                dataset_filename=
                    dataset_filename,
            )
        )


    executed = (
        execute_analysis_candidate(
            candidate,
            dataframe,
        )
    )


    inferential_status = (
        "executed"
        if (
            executed.execution_status
            ==
            "complete"
            and
            executed.statistical_result
            is not None
        )
        else "not_selected"
    )


    return (
        RequestedAnalysisExecution(
            request_id=
                request.request_id,

            request_text=
                request.request_text,

            kind=
                request.kind,

            plan_status=
                request.status,

            execution_status=
                executed.execution_status,

            inferential_status=
                inferential_status,

            source_filename=
                request.source_filename,

            source_locator=
                request.source_locator,

            evidence_quote=
                request.evidence_quote,

            dataset_id=
                dataset_id,

            dataset_filename=
                dataset_filename,

            analytical_grain=
                "event",

            analysis_mode=
                "exploratory",

            variables=
                variables,

            result=
                executed,

            warnings=[
                *executed.warnings,
            ],

            limitations=[
                *executed.limitations,
            ],

            executor_rule_version=
                REQUESTED_ANALYSIS_EXECUTOR_RULE_VERSION,
        )
    )


# ============================================================
# SINGLE REQUEST DISPATCH
# ============================================================

def execute_requested_analysis(
    *,
    request: RequestedAnalysisPlan,

    datasets: list[
        dict[
            str,
            Any,
        ]
    ],
) -> RequestedAnalysisExecution:
    # ========================================================
    # BLOCKED / AMBIGUOUS PLANS
    # ========================================================

    if (
        request.status
        !=
        "ready"
    ):
        return (
            not_executed_result(
                request
            )
        )


    # ========================================================
    # QUANTITATIVE CUSTOMER ASSOCIATIONS
    # ========================================================

    metric_column = (
        SUPPORTED_QUANTITATIVE_REQUESTS.get(
            request.kind
        )
    )


    if metric_column is not None:
        return (
            execute_quantitative_requested_analysis(
                request=
                    request,

                datasets=
                    datasets,

                metric_column=
                    metric_column,
            )
        )


    # ========================================================
    # BRIEF — DETERMINISTIC DESCRIPTIVE REQUESTS
    # ========================================================

    if (
        request.kind
        in {
            "revenue_moving_average",
            "revenue_by_category",
            "customers_by_period",
            "transaction_count",
            "products_sold_count",
            "top_products",
            "flop_products",
            "product_category_distribution",
            "lorenz_curve",
        }
    ):
        return (
            execute_brief_requested_analysis(
                request=
                    request,

                datasets=
                    datasets,
            )
        )


    # ========================================================
    # MULTI-TABLE DOCUMENTARY ASSOCIATIONS
    # ========================================================

    if (
        request.kind
        in {
            "gender_category_association",
            "age_category_association",
        }
    ):
        return (
            execute_context_requested_analysis(
                request=
                    request,

                datasets=
                    datasets,
            )
        )


    # ========================================================
    # UNSUPPORTED IN V0.6
    # ========================================================

    return (
        unsupported_result(
            request
        )
    )


# ============================================================
# REPORT
# ============================================================

def execute_requested_analysis_plan(
    *,
    plan: RequestedAnalysisPlanReport,

    datasets: list[
        dict[
            str,
            Any,
        ]
    ],
) -> RequestedAnalysisExecutionReport:
    results: list[
        RequestedAnalysisExecution
    ] = []


    for request in plan.requests:
        try:
            result = (
                execute_requested_analysis(
                    request=
                        request,

                    datasets=
                        datasets,
                )
            )


        except Exception as error:
            result = (
                RequestedAnalysisExecution(
                    request_id=
                        request.request_id,

                    request_text=
                        request.request_text,

                    kind=
                        request.kind,

                    plan_status=
                        request.status,

                    execution_status=
                        "failed",

                    inferential_status=
                        "not_selected",

                    source_filename=
                        request.source_filename,

                    source_locator=
                        request.source_locator,

                    evidence_quote=
                        request.evidence_quote,

                    warnings=[
                        (
                            "Le Requested Analysis "
                            "Executor a rencontré une "
                            "erreur inattendue."
                        ),
                        (
                            f"{type(error).__name__}: "
                            f"{error}"
                        ),
                    ],

                    executor_rule_version=
                        REQUESTED_ANALYSIS_EXECUTOR_RULE_VERSION,
                )
            )


        results.append(
            result
        )


    def count_status(
        status: str,
    ) -> int:
        return sum(
            1

            for result
            in results

            if (
                result.execution_status
                ==
                status
            )
        )


    def count_inferential_status(
        status: str,
    ) -> int:
        return sum(
            1

            for result
            in results

            if (
                result.inferential_status
                ==
                status
            )
        )


    attempted_count = sum(
        1

        for result
        in results

        if (
            result.execution_status
            not in {
                "not_executed",
                "not_supported_yet",
            }
        )
    )


    return (
        RequestedAnalysisExecutionReport(
            request_count=
                len(
                    results
                ),

            attempted_count=
                attempted_count,

            complete_count=
                count_status(
                    "complete"
                ),

            descriptive_only_count=
                count_status(
                    "descriptive_only"
                ),

            needs_information_count=
                count_status(
                    "needs_information"
                ),

            needs_specialized_method_count=
                count_status(
                    "needs_specialized_method"
                ),

            skipped_count=
                count_status(
                    "skipped"
                ),

            failed_count=
                count_status(
                    "failed"
                ),

            not_executed_count=
                count_status(
                    "not_executed"
                ),

            not_supported_yet_count=
                count_status(
                    "not_supported_yet"
                ),

            inference_executed_count=
                count_inferential_status(
                    "executed"
                ),

            inference_abstained_count=
                count_inferential_status(
                    "not_selected"
                ),

            results=
                results,

            executor_notes=[
                (
                    "Requested Analysis Executor "
                    "v0.6 prend en charge les "
                    "quatorze demandes "
                    "documentaires actuellement "
                    "au statut ready."
                ),
                (
                    "La seule demande non "
                    "exécutable du corpus Lapage "
                    "reste la répartition BtoB, "
                    "bloquée faute de variable "
                    "explicite permettant "
                    "d'identifier les clients BtoB."
                ),
                (
                    "Les tops et flops sont classés "
                    "selon le chiffre d'affaires "
                    "agrégé par référence, comme "
                    "spécifié par le Request "
                    "Planner."
                ),
                (
                    "DataLens v0.6 affiche dix "
                    "références par défaut pour "
                    "les classements ; cette limite "
                    "de présentation devra devenir "
                    "configurable."
                ),
                (
                    "La répartition par catégorie "
                    "compte les références produit "
                    "distinctes observées dans la "
                    "vue produit enrichie."
                ),
                (
                    "La courbe de Lorenz est "
                    "construite au grain client "
                    "à partir de total_spend, avec "
                    "tri croissant puis calcul des "
                    "parts cumulées de clients et "
                    "de chiffre d'affaires."
                ),
                (
                    "Le coefficient de Gini est "
                    "calculé comme résumé "
                    "descriptif de la concentration "
                    "sur les mêmes valeurs client."
                ),
                (
                    "Les agrégations monétaires "
                    "réutilisent les vues "
                    "analytiques déterministes du "
                    "Analytical View Builder au "
                    "lieu de recalculer des "
                    "jointures dans l'exécuteur."
                ),
                (
                    "La moyenne mobile v0.6 "
                    "réutilise la granularité "
                    "mensuelle et une fenêtre de "
                    "trois périodes ; ces paramètres "
                    "devront devenir configurables."
                ),
                (
                    "Les demandes genre × catégorie "
                    "et âge × catégorie utilisent "
                    "une vue événementielle "
                    "requested-only exclue de la "
                    "découverte exploratoire."
                ),
                (
                    "Lorsque le moteur refuse de "
                    "sélectionner un test de "
                    "corrélation, DataLens peut "
                    "fournir Pearson r et Spearman "
                    "rho comme coefficients "
                    "strictement descriptifs, sans "
                    "p-value."
                ),
            ],

            executor_rule_version=
                REQUESTED_ANALYSIS_EXECUTOR_RULE_VERSION,
        )
    )
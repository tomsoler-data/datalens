from __future__ import annotations

import re
import unicodedata

from app.discovery.schemas import (
    AnalysisDiscoveryReport,
    DiscoveredAnalysis,
    DiscoveredVariable,
)


# ============================================================
# SAFE DERIVED-METRIC UNITS
# ============================================================

SAFE_DIFFERENCE_UNITS = {
    "percentage",
    "currency",
    "duration",
    "age",
    "count",
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(
    value: str,
) -> str:
    text = (
        unicodedata
        .normalize(
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
        .lower()
        .strip()
    )


    text = re.sub(
        r"[^a-z0-9%$€£]+",
        "_",
        text,
    )


    return text.strip(
        "_"
    )


def text_tokens(
    value: str,
) -> set[
    str
]:
    return {
        token
        for token
        in normalize_text(
            value
        ).split(
            "_"
        )
        if token
    }


# ============================================================
# UNIT INFERENCE
# ============================================================

def infer_measure_unit(
    variable: DiscoveredVariable,
) -> str:
    """
    Infer only sufficiently explicit units.

    The validator intentionally prefers
    "unknown" rather than inventing a unit.
    """

    tokens = (
        text_tokens(
            variable.column
        )
    )


    # --------------------------------------------------------
    # PERCENTAGE
    # --------------------------------------------------------

    if (
        variable.semantic_role
        ==
        "percentage"
        or
        "%"
        in variable.column
        or
        bool(
            tokens
            &
            {
                "percent",
                "percentage",
                "pct",
                "pourcentage",
            }
        )
    ):
        return "percentage"


    # --------------------------------------------------------
    # RATE / RATIO
    # --------------------------------------------------------

    if (
        tokens
        &
        {
            "rate",
            "ratio",
            "taux",
        }
    ):
        return "rate"


    # --------------------------------------------------------
    # CURRENCY
    # --------------------------------------------------------

    if (
        "$"
        in variable.column
        or
        "€"
        in variable.column
        or
        "£"
        in variable.column
        or
        bool(
            tokens
            &
            {
                "price",
                "prix",
                "cost",
                "cout",
                "revenue",
                "revenu",
                "amount",
                "montant",
                "turnover",
                "sales",
                "ca",
            }
        )
    ):
        return "currency"


    # --------------------------------------------------------
    # DURATION
    # --------------------------------------------------------

    if (
        tokens
        &
        {
            "duration",
            "duree",
            "delay",
            "delai",
            "second",
            "seconds",
            "seconde",
            "secondes",
            "minute",
            "minutes",
            "hour",
            "hours",
            "heure",
            "heures",
            "day",
            "days",
            "jour",
            "jours",
        }
    ):
        return "duration"


    # --------------------------------------------------------
    # AGE
    # --------------------------------------------------------

    if (
        "age"
        in tokens
    ):
        return "age"


    # --------------------------------------------------------
    # EXPLICIT COUNTS
    # --------------------------------------------------------

    if (
        tokens
        &
        {
            "death",
            "deaths",
            "deces",
            "count",
            "counts",
            "nombre",
            "quantity",
            "quantite",
            "volume",
        }
    ):
        return "count"


    # --------------------------------------------------------
    # RAW POPULATION
    # --------------------------------------------------------

    if (
        "population"
        in tokens
        and
        "%"
        not in variable.column
        and
        "rate"
        not in tokens
        and
        "percent"
        not in tokens
        and
        "percentage"
        not in tokens
    ):
        return "count"


    return "unknown"


# ============================================================
# CONCEPT COMPATIBILITY
# ============================================================

def concept_overlap(
    left: DiscoveredVariable,
    right: DiscoveredVariable,
) -> set[
    str
]:
    return (
        set(
            left.concepts
        )
        &
        set(
            right.concepts
        )
    )


# ============================================================
# DERIVED GAP VALIDATION
# ============================================================

def derived_gap_is_semantically_valid(
    candidate: DiscoveredAnalysis,
) -> tuple[
    bool,
    str,
]:
    """
    Decide whether subtraction between two
    discovered measures is semantically safe.

    Examples:

        basic water % - safe water %
        -> potentially meaningful

        mortality rate - number of deaths
        -> invalid
    """

    if (
        len(
            candidate.variables
        )
        !=
        2
    ):
        return (
            False,
            (
                "A derived difference requires "
                "exactly two measures."
            ),
        )


    left = (
        candidate.variables[
            0
        ]
    )

    right = (
        candidate.variables[
            1
        ]
    )


    left_unit = (
        infer_measure_unit(
            left
        )
    )

    right_unit = (
        infer_measure_unit(
            right
        )
    )


    if (
        left_unit
        ==
        "unknown"
        or
        right_unit
        ==
        "unknown"
    ):
        return (
            False,
            (
                "The measurement units could "
                "not be established safely."
            ),
        )


    if (
        left_unit
        !=
        right_unit
    ):
        return (
            False,
            (
                "The two measures use "
                "incompatible units: "
                f"{left_unit} versus "
                f"{right_unit}."
            ),
        )


    if (
        left_unit
        not in
        SAFE_DIFFERENCE_UNITS
    ):
        return (
            False,
            (
                "This measurement unit is not "
                "currently approved for an "
                "automatic difference metric: "
                f"{left_unit}."
            ),
        )


    overlap = (
        concept_overlap(
            left,
            right,
        )
    )


    if not overlap:
        return (
            False,
            (
                "The two measures do not share "
                "enough semantic context to "
                "justify an automatic "
                "difference."
            ),
        )


    return (
        True,
        (
            "The measures have compatible "
            f"{left_unit} units and share "
            "semantic context."
        ),
    )


# ============================================================
# GROUP COMPARISON VALIDATION
# ============================================================

def group_comparison_is_valid(
    candidate: DiscoveredAnalysis,
) -> tuple[
    bool,
    str,
]:
    valid_group_count = (
        candidate
        .observed_signals
        .get(
            "valid_group_count"
        )
    )


    if (
        valid_group_count
        is None
    ):
        return (
            True,
            (
                "No contradictory group-count "
                "signal was recorded."
            ),
        )


    if (
        int(
            valid_group_count
        )
        <
        2
    ):
        return (
            False,
            (
                "Fewer than two groups contain "
                "valid observations for the "
                "measure."
            ),
        )


    return (
        True,
        (
            "At least two groups contain "
            "valid observations."
        ),
    )


# ============================================================
# CROSS-DATASET ANNOTATION
# ============================================================

def annotate_cross_dataset_candidate(
    candidate: DiscoveredAnalysis,
) -> None:
    if (
        candidate.scope
        !=
        "cross_dataset"
    ):
        return


    signals = (
        candidate.observed_signals
    )


    # --------------------------------------------------------
    # DIRECT
    # --------------------------------------------------------

    if (
        candidate.relationship_status
        ==
        "validated"
    ):
        signals[
            "join_safety"
        ] = "direct"

        signals[
            "automatic_execution_allowed"
        ] = True

        return


    # --------------------------------------------------------
    # MATCHED SUBSET
    # --------------------------------------------------------

    if (
        candidate.relationship_status
        ==
        "partial"
    ):
        signals[
            "join_safety"
        ] = "matched_subset"

        signals[
            "automatic_execution_allowed"
        ] = True


        limitation_already_present = any(
            (
                "couverture"
                in limitation.lower()
            )
            or
            (
                "exclu"
                in limitation.lower()
            )
            for limitation
            in candidate.limitations
        )


        if (
            not limitation_already_present
        ):
            candidate.limitations.append(
                (
                    "L'analyse devra conserver "
                    "la couverture du sous-ensemble "
                    "apparié afin d'éviter de "
                    "présenter celui-ci comme "
                    "représentatif de l'ensemble "
                    "des données."
                )
            )


        return


    # --------------------------------------------------------
    # GRAIN ALIGNMENT
    # --------------------------------------------------------

    signals[
        "join_safety"
    ] = (
        "grain_alignment_required"
    )

    signals[
        "automatic_execution_allowed"
    ] = False


# ============================================================
# CANDIDATE VALIDATION
# ============================================================

def validate_candidate(
    candidate: DiscoveredAnalysis,
) -> tuple[
    bool,
    str,
]:
    if (
        candidate.family
        ==
        "derived_gap"
    ):
        return (
            derived_gap_is_semantically_valid(
                candidate
            )
        )


    if (
        candidate.family
        ==
        "group_comparison"
    ):
        return (
            group_comparison_is_valid(
                candidate
            )
        )


    return (
        True,
        (
            "No blocking semantic issue "
            "was detected."
        ),
    )


# ============================================================
# COMPLETE DISCOVERY VALIDATION
# ============================================================

def validate_discovery_report(
    report: AnalysisDiscoveryReport,
) -> AnalysisDiscoveryReport:
    """
    Apply the semantic validation stage after
    broad analysis discovery.

    Discovery should have high recall.
    Validation should be conservative.
    """

    validated_candidates: list[
        DiscoveredAnalysis
    ] = []


    rejected_candidates: list[
        dict[
            str,
            str,
        ]
    ] = []


    for candidate in (
        report.candidates
    ):
        (
            valid,
            reason,
        ) = validate_candidate(
            candidate
        )


        if not valid:
            rejected_candidates.append(
                {
                    "analysis_id":
                        candidate.analysis_id,

                    "title":
                        candidate.title,

                    "reason":
                        reason,
                }
            )

            continue


        candidate.observed_signals[
            "semantic_validation"
        ] = {
            "status":
                "passed",

            "reason":
                reason,
        }


        annotate_cross_dataset_candidate(
            candidate
        )


        validated_candidates.append(
            candidate
        )


    # ========================================================
    # SORT AGAIN AFTER VALIDATION
    # ========================================================

    validated_candidates.sort(
        key=lambda candidate: (
            candidate.priority_score,

            1
            if (
                candidate.scope
                ==
                "cross_dataset"
            )
            else
            0,
        ),
        reverse=True,
    )


    # ========================================================
    # UPDATE COUNTS
    # ========================================================

    report.candidates = (
        validated_candidates
    )


    report.candidate_count = len(
        validated_candidates
    )


    report.single_dataset_candidate_count = sum(
        1
        for candidate
        in validated_candidates
        if (
            candidate.scope
            ==
            "single_dataset"
        )
    )


    report.cross_dataset_candidate_count = sum(
        1
        for candidate
        in validated_candidates
        if (
            candidate.scope
            ==
            "cross_dataset"
        )
    )


    # ========================================================
    # TRACEABILITY
    # ========================================================

    report.discovery_notes.append(
        (
            "A semantic validation pass was "
            "applied after broad candidate "
            "generation."
        )
    )


    report.discovery_notes.append(
        (
            "Candidates with incompatible "
            "measurement units or invalid "
            "group structures are removed "
            "before execution."
        )
    )


    report.discovery_notes.append(
        (
            f"{len(rejected_candidates)} "
            "candidate(s) were rejected by "
            "semantic validation."
        )
    )


    for rejected in (
        rejected_candidates
    ):
        report.discovery_notes.append(
            (
                "Rejected candidate — "
                f"{rejected['title']}: "
                f"{rejected['reason']}"
            )
        )


    report.discovery_rule_version = (
        "analysis_discovery_v0.3"
    )


    return report
from typing import Any

from app.discovery.engine import (
    discover_analyses as
    discover_analyses_unvalidated,
)

from app.discovery.schemas import (
    AnalysisDiscoveryReport,
    AnalysisScope,
    DiscoveredAnalysis,
    DiscoveredVariable,
    DiscoveryReadiness,
    RelationshipStatus,
    RelationshipSummary,
)

from app.discovery.validator import (
    derived_gap_is_semantically_valid,
    infer_measure_unit,
    validate_candidate,
    validate_discovery_report,
)


# ============================================================
# PUBLIC DISCOVERY ENTRY POINT
# ============================================================

def discover_analyses(
    *,
    datasets: list[
        dict[
            str,
            Any,
        ]
    ],
    objective: str | None = None,
) -> AnalysisDiscoveryReport:
    """
    Public DataLens discovery entry point.

    The discovery engine deliberately generates
    a broad catalog of potentially interesting
    analyses.

    The semantic validator then removes
    candidates that should not continue toward
    execution, for example:

    - differences between incompatible units;
    - invalid group comparisons;
    - semantically unsafe derived metrics.

    This separation keeps candidate generation
    broad while keeping execution conservative.
    """

    report = (
        discover_analyses_unvalidated(
            datasets=
                datasets,

            objective=
                objective,
        )
    )


    validated_report = (
        validate_discovery_report(
            report
        )
    )


    return validated_report


__all__ = [
    "AnalysisDiscoveryReport",
    "AnalysisScope",
    "DiscoveredAnalysis",
    "DiscoveredVariable",
    "DiscoveryReadiness",
    "RelationshipStatus",
    "RelationshipSummary",
    "derived_gap_is_semantically_valid",
    "discover_analyses",
    "infer_measure_unit",
    "validate_candidate",
    "validate_discovery_report",
]
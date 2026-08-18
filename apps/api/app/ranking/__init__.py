from __future__ import annotations

from importlib import (
    import_module,
)

from typing import (
    Any,
)


# ============================================================
# PUBLIC API
# ============================================================
#
# The ranking package deliberately uses lazy imports.
#
# This prevents package-initialization cycles such as:
#
# app.ranking.__init__
#     -> app.ranking.unified
#         -> app.ranking.unified_schemas
#
# Submodules are imported only when a public attribute is
# actually requested.
# ============================================================


_EXPORTS: dict[
    str,
    tuple[
        str,
        str,
    ],
] = {
    # ========================================================
    # LEGACY / CROSS-DATASET RANKER
    # ========================================================

    "AnalysisRankingReport": (
        "app.ranking.schemas",
        "AnalysisRankingReport",
    ),

    "AssociationDirection": (
        "app.ranking.schemas",
        "AssociationDirection",
    ),

    "AssociationStrength": (
        "app.ranking.schemas",
        "AssociationStrength",
    ),

    "FindingTier": (
        "app.ranking.schemas",
        "FindingTier",
    ),

    "RankedAnalysis": (
        "app.ranking.schemas",
        "RankedAnalysis",
    ),

    "calculate_interestingness_score": (
        "app.ranking.ranker",
        "calculate_interestingness_score",
    ),

    "rank_cross_dataset_execution": (
        "app.ranking.ranker",
        "rank_cross_dataset_execution",
    ),

    "rank_cross_dataset_result": (
        "app.ranking.ranker",
        "rank_cross_dataset_result",
    ),


    # ========================================================
    # UNIFIED SCHEMAS
    # ========================================================

    "UnifiedFindingTier": (
        "app.ranking.unified_schemas",
        "UnifiedFindingTier",
    ),

    "UnifiedRankedAnalysis": (
        "app.ranking.unified_schemas",
        "UnifiedRankedAnalysis",
    ),

    "UnifiedRankingReport": (
        "app.ranking.unified_schemas",
        "UnifiedRankingReport",
    ),

    "UnifiedSignalType": (
        "app.ranking.unified_schemas",
        "UnifiedSignalType",
    ),


    # ========================================================
    # UNIFIED RANKER
    # ========================================================

    "rank_cross_result": (
        "app.ranking.unified",
        "rank_cross_result",
    ),

    "rank_single_result": (
        "app.ranking.unified",
        "rank_single_result",
    ),

    "rank_unified_analysis": (
        "app.ranking.unified",
        "rank_unified_analysis",
    ),
}


__all__ = list(
    _EXPORTS.keys()
)


# ============================================================
# LAZY ATTRIBUTE RESOLUTION
# ============================================================

def __getattr__(
    name: str,
) -> Any:
    target = (
        _EXPORTS.get(
            name
        )
    )


    if target is None:
        raise AttributeError(
            (
                f"module 'app.ranking' "
                f"has no attribute {name!r}"
            )
        )


    module_name, attribute_name = (
        target
    )


    module = import_module(
        module_name
    )


    value = getattr(
        module,
        attribute_name,
    )


    # Cache the resolved attribute in the package
    # namespace so subsequent accesses do not
    # repeat the import lookup.
    globals()[
        name
    ] = value


    return value


# ============================================================
# DIRECTORY SUPPORT
# ============================================================

def __dir__() -> list[
    str
]:
    return sorted(
        set(
            globals()
        )
        |
        set(
            __all__
        )
    )
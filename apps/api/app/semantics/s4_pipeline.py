from __future__ import annotations

from typing import (
    Any,
)

from app.ai.provider import (
    DEFAULT_MODEL,
)

from app.semantics.family import (
    build_quantity_family_reports,
)

from app.semantics.pipeline import (
    prepare_datasets_semantics,
)

from app.semantics.s4_pipeline_schemas import (
    SemanticPreparationS4Result,
)


# ============================================================
# VERSION
# ============================================================

S4_PREPARATION_RULE_VERSION = (
    "semantic_preparation_s4_1_v0.1"
)


# ============================================================
# RESULT BUILDER
# ============================================================

def build_s4_preparation_result(
    *,
    base_preparation,
    quantity_family_reports,
) -> SemanticPreparationS4Result:
    return SemanticPreparationS4Result(
        base_preparation=
            base_preparation,

        quantity_family_dataset_count=
            len(
                quantity_family_reports
            ),

        quantity_family_eligible_column_count=
            sum(
                report.eligible_column_count

                for report
                in quantity_family_reports
            ),

        quantity_family_assignment_count=
            sum(
                report.assignment_count

                for report
                in quantity_family_reports
            ),

        quantity_family_llm_assignment_count=
            sum(
                report.llm_assignment_count

                for report
                in quantity_family_reports
            ),

        quantity_family_fallback_assignment_count=
            sum(
                report.fallback_assignment_count

                for report
                in quantity_family_reports
            ),

        quantity_family_clustering_failure_count=
            sum(
                not report.clustering_succeeded

                for report
                in quantity_family_reports
            ),

        quantity_family_reports=
            quantity_family_reports,

        semantic_rule_version=
            S4_PREPARATION_RULE_VERSION,
    )


# ============================================================
# MULTI-DATASET S4.1 PREPARATION
#
# Frozen S3 preparation still executes first.
#
# S4.1 expands quantity-family semantic coverage without
# mutating the S3 profiles or comparator.
# ============================================================

def prepare_datasets_semantics_s4(
    *,
    datasets: list[
        dict[
            str,
            Any,
        ]
    ],
    model: str = DEFAULT_MODEL,
) -> SemanticPreparationS4Result:
    base_preparation = (
        prepare_datasets_semantics(
            datasets=
                datasets,

            model=
                model,
        )
    )


    quantity_family_reports = (
        build_quantity_family_reports(
            profiles=
                base_preparation.profiles,

            model=
                model,
        )
    )


    return build_s4_preparation_result(
        base_preparation=
            base_preparation,

        quantity_family_reports=
            quantity_family_reports,
    )


# ============================================================
# SINGLE-DATASET S4.1 PREPARATION
# ============================================================

def prepare_dataset_semantics_s4(
    *,
    dataset_id: str,
    filename: str,
    dataframe: Any,
    model: str = DEFAULT_MODEL,
) -> SemanticPreparationS4Result:
    return prepare_datasets_semantics_s4(
        datasets=[
            {
                "dataset_id":
                    dataset_id,

                "filename":
                    filename,

                "dataframe":
                    dataframe,
            }
        ],

        model=
            model,
    )

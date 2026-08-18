from __future__ import annotations

from typing import (
    Any,
)

from app.ai.provider import (
    DEFAULT_MODEL,
)

from app.semantics.normalizer import (
    normalize_dataset_semantics,
)

from app.semantics.pipeline_schemas import (
    SemanticDatasetNormalizationAudit,
    SemanticNormalizationChange,
    SemanticPreparationResult,
)

from app.semantics.profiler import (
    profile_dataset_semantics,
    profile_datasets_semantics,
)

from app.semantics.schemas import (
    DatasetSemanticProfile,
)


# ============================================================
# AUDITED FIELDS
# ============================================================

AUDITED_FIELDS = (
    "concept",
    "domain",
    "semantic_group",
    "variant",
    "measure_kind",
    "unit_kind",
)


# ============================================================
# RULE IDENTIFICATION
# ============================================================

def normalization_rule_for(
    *,
    field: str,
    before: str,
    after: str,
) -> str:
    if (
        field
        ==
        "semantic_group"
    ):
        return (
            "shared_lexical_semantic_group"
        )


    if (
        field
        ==
        "variant"
    ):
        if (
            after
            ==
            "unknown"
        ):
            return (
                "unsupported_variant_removed"
            )


        return (
            "explicit_literal_variant"
        )


    if (
        field
        in {
            "measure_kind",
            "unit_kind",
        }
    ):
        return (
            "deterministic_measure_unit_signal"
        )


    return (
        "deterministic_semantic_normalization"
    )


# ============================================================
# DATASET AUDIT
# ============================================================

def build_normalization_audit(
    *,
    raw_profile: DatasetSemanticProfile,
    normalized_profile: DatasetSemanticProfile,
) -> SemanticDatasetNormalizationAudit:
    raw_lookup = {
        profile.column:
            profile

        for profile
        in raw_profile.columns
    }


    normalized_lookup = {
        profile.column:
            profile

        for profile
        in normalized_profile.columns
    }


    changes: list[
        SemanticNormalizationChange
    ] = []


    changed_columns: set[
        str
    ] = set()


    for column, raw_column in (
        raw_lookup.items()
    ):
        normalized_column = (
            normalized_lookup.get(
                column
            )
        )


        if (
            normalized_column
            is None
        ):
            continue


        for field in (
            AUDITED_FIELDS
        ):
            before = str(
                getattr(
                    raw_column,
                    field,
                )
            )


            after = str(
                getattr(
                    normalized_column,
                    field,
                )
            )


            if (
                before
                ==
                after
            ):
                continue


            changed_columns.add(
                column
            )


            changes.append(
                SemanticNormalizationChange(
                    dataset_id=
                        raw_profile.dataset_id,

                    filename=
                        raw_profile.filename,

                    column=
                        column,

                    field=
                        field,

                    before=
                        before,

                    after=
                        after,

                    rule=
                        normalization_rule_for(
                            field=
                                field,

                            before=
                                before,

                            after=
                                after,
                        ),
                )
            )


    return (
        SemanticDatasetNormalizationAudit(
            dataset_id=
                raw_profile.dataset_id,

            filename=
                raw_profile.filename,

            column_count=
                len(
                    raw_profile.columns
                ),

            changed_column_count=
                len(
                    changed_columns
                ),

            change_count=
                len(
                    changes
                ),

            normalization_applied=
                bool(
                    changes
                ),

            changes=
                changes,
        )
    )


# ============================================================
# SINGLE DATASET PREPARATION
# ============================================================

def prepare_dataset_semantics(
    *,
    dataset_id: str,
    filename: str,
    dataframe: Any,
    model: str = DEFAULT_MODEL,
) -> tuple[
    DatasetSemanticProfile,
    SemanticDatasetNormalizationAudit,
]:
    raw_profile = (
        profile_dataset_semantics(
            dataset_id=
                dataset_id,

            filename=
                filename,

            dataframe=
                dataframe,

            model=
                model,
        )
    )


    normalized_profile = (
        normalize_dataset_semantics(
            raw_profile
        )
    )


    audit = (
        build_normalization_audit(
            raw_profile=
                raw_profile,

            normalized_profile=
                normalized_profile,
        )
    )


    return (
        normalized_profile,
        audit,
    )


# ============================================================
# PREPARE EXISTING RAW PROFILES
# ============================================================

def normalize_semantic_profiles_with_audit(
    *,
    raw_profiles: list[
        DatasetSemanticProfile
    ],
) -> SemanticPreparationResult:
    profiles: list[
        DatasetSemanticProfile
    ] = []


    audits: list[
        SemanticDatasetNormalizationAudit
    ] = []


    for raw_profile in (
        raw_profiles
    ):
        normalized_profile = (
            normalize_dataset_semantics(
                raw_profile
            )
        )


        audit = (
            build_normalization_audit(
                raw_profile=
                    raw_profile,

                normalized_profile=
                    normalized_profile,
            )
        )


        profiles.append(
            normalized_profile
        )


        audits.append(
            audit
        )


    return build_preparation_result(
        profiles=
            profiles,

        audits=
            audits,
    )


# ============================================================
# RESULT BUILDER
# ============================================================

def build_preparation_result(
    *,
    profiles: list[
        DatasetSemanticProfile
    ],
    audits: list[
        SemanticDatasetNormalizationAudit
    ],
) -> SemanticPreparationResult:
    column_count = sum(
        len(
            profile.columns
        )
        for profile
        in profiles
    )


    normalized_dataset_count = sum(
        audit.normalization_applied
        for audit
        in audits
    )


    normalized_columns: set[
        tuple[
            str,
            str,
        ]
    ] = set()


    normalization_change_count = 0


    for audit in audits:
        normalization_change_count += (
            audit.change_count
        )


        for change in (
            audit.changes
        ):
            normalized_columns.add(
                (
                    change.dataset_id,
                    change.column,
                )
            )


    return (
        SemanticPreparationResult(
            dataset_count=
                len(
                    profiles
                ),

            column_count=
                column_count,

            normalized_dataset_count=
                normalized_dataset_count,

            normalized_column_count=
                len(
                    normalized_columns
                ),

            normalization_change_count=
                normalization_change_count,

            profiles=
                profiles,

            audits=
                audits,
        )
    )


# ============================================================
# MULTI-DATASET PREPARATION
# ============================================================

def prepare_datasets_semantics(
    *,
    datasets: list[
        dict[
            str,
            Any,
        ]
    ],
    model: str = DEFAULT_MODEL,
) -> SemanticPreparationResult:
    raw_profiles = (
        profile_datasets_semantics(
            datasets=
                datasets,

            model=
                model,
        )
    )


    return (
        normalize_semantic_profiles_with_audit(
            raw_profiles=
                raw_profiles,
        )
    )

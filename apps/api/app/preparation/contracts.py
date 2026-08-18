from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ============================================================
# PREPARATION STAGES
# ============================================================


class PreparationStage(str, Enum):
    UNDERSTANDING = "understanding"
    QUALITY = "quality"
    PLANNING = "planning"
    CLEANING = "cleaning"
    TRANSFORMATION = "transformation"
    COMBINATION = "combination"
    VALIDATION = "validation"


# ============================================================
# LEGACY / PROTOTYPE QUALITY CONTRACTS
#
# Conservés temporairement pour que quality_review.py v0.1
# reste importable.
#
# Le pipeline de production doit utiliser :
#
# app.preparation.data_quality.DataQualityReport
#
# et non QualityReviewResult.
# ============================================================


class IssueSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ColumnDataType(str, Enum):
    NUMERIC = "numeric"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    CATEGORICAL = "categorical"
    TEXT = "text"
    UNKNOWN = "unknown"


class QualityIssue(BaseModel):
    code: str

    severity: IssueSeverity

    stage: PreparationStage = PreparationStage.QUALITY

    column: Optional[str] = None

    message: str

    evidence: Dict[str, Any] = Field(
        default_factory=dict
    )

    suggested_action: Optional[str] = None


class ColumnQualityProfile(BaseModel):
    name: str

    pandas_dtype: str

    inferred_type: ColumnDataType

    row_count: int

    non_null_count: int

    missing_count: int

    missing_rate: float

    unique_count: int

    unique_rate: float

    identifier_candidate: bool = False

    empty_string_count: int = 0

    whitespace_issue_count: int = 0

    potential_outlier_count: int = 0

    numeric_coercion_rate: Optional[float] = None

    sample_values: List[Any] = Field(
        default_factory=list
    )


class QualityReviewResult(BaseModel):
    stage: PreparationStage = PreparationStage.QUALITY

    row_count: int

    column_count: int

    missing_cells: int

    missing_rate: float

    duplicate_rows: int

    blocking_issue_count: int

    warning_count: int

    info_count: int

    ready_for_cleaning: bool

    columns: List[ColumnQualityProfile] = Field(
        default_factory=list
    )

    issues: List[QualityIssue] = Field(
        default_factory=list
    )


# ============================================================
# PREPARATION PLANNER
# ============================================================


class DecisionStatus(str, Enum):
    """
    État d'une décision de préparation.

    AUTO_APPROVABLE
        Une transformation déterministe à faible risque
        peut être préparée automatiquement.

    REVIEW_REQUIRED
        Une décision humaine est nécessaire avant exécution.

    NEEDS_CONTEXT
        DataLens ne possède pas encore suffisamment de contexte
        pour prendre une décision défendable.
    """

    AUTO_APPROVABLE = "auto_approvable"
    REVIEW_REQUIRED = "review_required"
    NEEDS_CONTEXT = "needs_context"


class DecisionRisk(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PreparationAction(str, Enum):
    """
    Représentation normalisée des grandes familles
    d'actions de préparation.

    Le DataQualityReport conserve également l'opération
    technique originale produite par le moteur déterministe.
    """

    KEEP_AS_IS = "keep_as_is"

    # --------------------------------------------------------
    # Strings / categories
    # --------------------------------------------------------

    TRIM_WHITESPACE = "trim_whitespace"

    NORMALIZE_EMPTY_TO_MISSING = (
        "normalize_empty_to_missing"
    )

    NORMALIZE_MISSING_MARKERS = (
        "normalize_missing_markers"
    )

    NORMALIZE_CASE = "normalize_case"

    MERGE_CATEGORY_VALUES = (
        "merge_category_values"
    )

    KEEP_CATEGORIES_SEPARATE = (
        "keep_categories_separate"
    )

    # --------------------------------------------------------
    # Types
    # --------------------------------------------------------

    CONVERT_TO_NUMERIC = (
        "convert_to_numeric"
    )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    KEEP_MISSING = "keep_missing"

    DROP_ROWS_WITH_MISSING = (
        "drop_rows_with_missing"
    )

    DROP_COLUMN = "drop_column"

    IMPUTE_MEAN = "impute_mean"

    IMPUTE_MEDIAN = "impute_median"

    IMPUTE_MODE = "impute_mode"

    CREATE_MISSING_CATEGORY = (
        "create_missing_category"
    )

    DOMAIN_SPECIFIC_VALUE = (
        "domain_specific_value"
    )

    # --------------------------------------------------------
    # Duplicates
    # --------------------------------------------------------

    REVIEW_DUPLICATES = (
        "review_duplicates"
    )

    KEEP_DUPLICATE_ROWS = (
        "keep_duplicate_rows"
    )

    REMOVE_DUPLICATE_ROWS = (
        "remove_duplicate_rows"
    )

    # --------------------------------------------------------
    # Outliers / invalid values
    # --------------------------------------------------------

    INVESTIGATE_OUTLIERS = (
        "investigate_outliers"
    )

    CAP_OUTLIERS = "cap_outliers"

    REMOVE_OUTLIER_ROWS = (
        "remove_outlier_rows"
    )

    REVIEW_INVALID_VALUES = (
        "review_invalid_values"
    )

    REVIEW_INVALID_DATES = (
        "review_invalid_dates"
    )

    # --------------------------------------------------------
    # Structure
    # --------------------------------------------------------

    RENAME_DUPLICATE_COLUMNS = (
        "rename_duplicate_columns"
    )

    REIMPORT_OR_FIX_SOURCE = (
        "reimport_or_fix_source"
    )

    CONFIRM_IDENTIFIER = (
        "confirm_identifier"
    )

    # --------------------------------------------------------
    # Semantic / contextual
    # --------------------------------------------------------

    REVIEW_SEMANTIC_CONTEXT = (
        "review_semantic_context"
    )


class PreparationDecision(BaseModel):
    """
    Décision unifiée construite à partir de :

    - DataQualityReport ;
    - SemanticReviewReport, lorsqu'une décision sémantique
      validée existe pour le même issue_id.

    Le planner ne modifie jamais les données.
    """

    decision_id: str

    stage: PreparationStage = (
        PreparationStage.PLANNING
    )

    # --------------------------------------------------------
    # Provenance qualité
    # --------------------------------------------------------

    source_issue_id: str

    source_issue_kind: str

    dataset_id: str

    dataset_filename: str

    column: Optional[str] = None

    severity: str

    title: str

    status: DecisionStatus

    risk: DecisionRisk

    rationale: str

    evidence: Dict[str, Any] = Field(
        default_factory=dict
    )

    # --------------------------------------------------------
    # Proposition déterministe originale
    # --------------------------------------------------------

    source_operation: Optional[str] = None

    source_operation_description: Optional[str] = None

    source_operation_parameters: Dict[
        str,
        Any,
    ] = Field(
        default_factory=dict
    )

    source_automatic_safe: bool = False

    source_requires_user_confirmation: bool = True

    # --------------------------------------------------------
    # Enrichissement sémantique éventuel
    # --------------------------------------------------------

    semantic_verdict: Optional[str] = None

    semantic_confidence: Optional[float] = None

    semantic_rationale: Optional[str] = None

    semantic_user_message: Optional[str] = None

    semantic_source_values: List[str] = Field(
        default_factory=list
    )

    semantic_canonical_value: Optional[str] = None

    semantic_python_validated: bool = False

    semantic_executable: bool = False

    semantic_validation_notes: List[str] = Field(
        default_factory=list
    )

    # --------------------------------------------------------
    # Decision support
    # --------------------------------------------------------

    context_required: List[str] = Field(
        default_factory=list
    )

    candidate_actions: List[
        PreparationAction
    ] = Field(
        default_factory=list
    )

    recommended_action: Optional[
        PreparationAction
    ] = None

    selected_action: Optional[
        PreparationAction
    ] = None

    # L'opération technique peut être conservée même lorsqu'elle
    # n'a pas encore de PreparationAction normalisée.
    recommended_operation: Optional[str] = None

    selected_operation: Optional[str] = None

    requires_human_validation: bool = True


class PreparationPlan(BaseModel):
    """
    Rapport de planification de la préparation.

    Aucune transformation n'est exécutée par ce contrat.
    """

    stage: PreparationStage = (
        PreparationStage.PLANNING
    )

    quality_issue_count: int

    semantic_decision_count: int

    semantic_enriched_count: int

    total_decisions: int

    auto_approvable_count: int

    review_required_count: int

    needs_context_count: int

    unresolved_count: int

    ready_for_execution: bool

    decisions: List[
        PreparationDecision
    ] = Field(
        default_factory=list
    )

    notes: List[str] = Field(
        default_factory=list
    )

    rule_version: str = (
        "preparation_planner_v0.2"
    )
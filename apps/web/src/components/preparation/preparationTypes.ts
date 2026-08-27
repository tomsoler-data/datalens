export type PreparationStage =
  | "import"
  | "understand"
  | "quality"
  | "clean"
  | "transform"
  | "combine"
  | "validate";


export type PreparationStageStatus =
  | "not_started"
  | "review_required"
  | "blocked"
  | "passed"
  | "skipped";


export type PreparationStageRecord = {
  stage:
    PreparationStage;

  status:
    PreparationStageStatus;

  required:
    boolean;

  dataset_ids:
    string[];

  evidence_refs:
    string[];

  blocking_reasons:
    string[];

  details:
    Record<
      string,
      unknown
    >;
};


export type PreparationWorkflowSnapshot = {
  workflow_id:
    string;

  stage_count:
    number;

  resolved_stage_count:
    number;

  passed_stage_count:
    number;

  skipped_stage_count:
    number;

  review_required_count:
    number;

  blocked_stage_count:
    number;

  not_started_count:
    number;

  /*
   * Immutable Preparation roots.
   */
  selected_analysis_dataset_ids:
    string[];

  /*
   * Final materialized datasets explicitly selected
   * for analytical execution.
   */
  analysis_output_dataset_ids:
    string[];

  /*
   * Final analytical outputs certified by VALIDATE.
   */
  validated_analysis_dataset_ids:
    string[];

  next_stage:
    PreparationStage |
    null;

  ready_for_analysis:
    boolean;

  blocking_reasons:
    string[];

  stages:
    PreparationStageRecord[];

  notes:
    string[];

  rule_version:
    string;
};


export type PreparationSessionView = {
  session_version:
    string;

  workflow_id:
    string;

  revision:
    number;

  /*
   * Immutable Preparation roots.
   */
  selected_analysis_dataset_ids:
    string[];

  /*
   * Final analytical output scope.
   */
  analysis_output_dataset_ids:
    string[];

  snapshot:
    PreparationWorkflowSnapshot;
};


/* ============================================================
   WORKFLOW HISTORY
   PREPARATION_WORKFLOW_HISTORY_FRONTEND_V0_1
============================================================ */


/* ============================================================
   WORKFLOW METADATA
   PREPARATION_WORKFLOW_METADATA_FRONTEND_V0_1
============================================================ */


export type PreparationSessionCatalogItem = {
  session:
    PreparationSessionView;

  display_name:
    string;

  name_source:
    "user" |
    "automatic";

  created_at_utc:
    string;

  updated_at_utc:
    string;

  archived:
    boolean;

  archived_at_utc:
    string |
    null;
};


export type PreparationSessionCatalogResponse = {
  count:
    number;

  sessions:
    PreparationSessionCatalogItem[];

  api_version:
    string;
};



export type PreparationSessionCapabilities = {
  api_version:
    string;

  session_version:
    string;

  storage:
    string;

  persistent:
    boolean;

  client_can_set_workflow_id:
    boolean;

  client_can_update_stage_status:
    boolean;

  client_can_set_ready_for_analysis:
    boolean;

  client_can_select_analysis_output:
    boolean;

  notes:
    string[];
};


export type CreatePreparationSessionRequest = {
  selected_analysis_dataset_ids:
    string[];

  display_name?:
    string |
    null;
};


export type PreparationAnalysisOutputCandidate = {
  dataset_id:
    string;

  dataset_filename:
    string;

  stage:
    "source" |
    "clean" |
    "transform" |
    "combine" |
    string;

  rows:
    number;

  columns:
    number;

  parent_dataset_ids:
    string[];

  evidence_refs:
    string[];

  is_root_dataset:
    boolean;

  is_selected:
    boolean;

  is_validated:
    boolean;
};


export type PreparationAnalysisOutputCandidatesResponse = {
  workflow_id:
    string;

  revision:
    number;

  selected_analysis_dataset_ids:
    string[];

  analysis_output_dataset_ids:
    string[];

  validated_analysis_dataset_ids:
    string[];

  locked:
    boolean;

  candidate_count:
    number;

  candidates:
    PreparationAnalysisOutputCandidate[];

  api_version:
    string;
};




// ============================================================
// DATASET IDENTITY
// ============================================================


export type DatasetIdentityStatus =
  | "single_key"
  | "composite_key"
  | "surrogate_recommended";


export type DatasetIdentityCandidateKind =
  | "single"
  | "composite";


export type DatasetIdentityCandidate = {
  columns:
    string[];

  kind:
    DatasetIdentityCandidateKind;

  row_count:
    number;

  unique_count:
    number;

  missing_row_count:
    number;

  uniqueness_ratio:
    number;

  complete:
    boolean;

  unique:
    boolean;

  identifier_name_signal:
    boolean;

  deterministic_score:
    number;

  rationale:
    string[];
};


export type DatasetIdentityReport = {
  dataset_id:
    string;

  dataset_filename:
    string;

  row_count:
    number;

  column_count:
    number;

  status:
    DatasetIdentityStatus;

  preferred_candidate:
    DatasetIdentityCandidate |
    null;

  candidates:
    DatasetIdentityCandidate[];

  mechanically_unique_columns:
    string[];

  identifier_like_columns:
    string[];

  surrogate_key_recommended:
    boolean;

  suggested_surrogate_column:
    string |
    null;

  reasons:
    string[];

  rule_version:
    string;
};


export type DatasetIdentityExplanationAction =
  | "keep_detected_key"
  | "create_surrogate_key"
  | "review_identity";


export type DatasetIdentityExplanation = {
  dataset_id:
    string;

  dataset_filename:
    string;

  deterministic_status:
    string;

  action:
    DatasetIdentityExplanationAction;

  confidence:
    number;

  title:
    string;

  explanation:
    string;

  user_message:
    string;

  referenced_columns:
    string[];

  surrogate_column:
    string |
    null;

  cautions:
    string[];

  python_validated:
    boolean;

  requires_user_confirmation:
    boolean;

  executable:
    boolean;

  model:
    string;

  rule_version:
    string;
};


export type PreparationIdentityInspectResponse = {
  workflow_id:
    string;

  dataset_id:
    string;

  dataset_filename:
    string;

  artifact_stage:
    string;

  report:
    DatasetIdentityReport;

  explanation:
    DatasetIdentityExplanation |
    null;

  ai_error:
    string |
    null;

  surrogate_request_id:
    string |
    null;

  identity_resolved:
    boolean;

  resolution_kind:
    string |
    null;

  can_create_surrogate:
    boolean;

  can_continue_without_surrogate:
    boolean;

  mutation_locked:
    boolean;

  mutation_lock_reason:
    string |
    null;

  api_version:
    string;

  identity_rule_version:
    string;

  explanation_rule_version:
    string;

  artifact_store_version:
    string;
};


export type PreparationIdentityContinueResponse = {
  workflow_id:
    string;

  dataset_id:
    string;

  dataset_filename:
    string;

  request_id:
    string;

  resolution_kind:
    string;

  identity_resolved:
    boolean;

  api_version:
    string;

  identity_resolution_version:
    string;
};


export type PreparationIdentityCreateSurrogateResponse = {
  workflow_id:
    string;

  source_dataset_id:
    string;

  source_dataset_filename:
    string;

  output_dataset_id:
    string;

  output_dataset_filename:
    string;

  surrogate_column:
    string;

  rows:
    number;

  columns:
    number;

  parent_dataset_ids:
    string[];

  report_before:
    DatasetIdentityReport;

  session:
    PreparationSessionView;

  api_version:
    string;

  identity_rule_version:
    string;

  artifact_store_version:
    string;
};




// ============================================================
// ANALYSIS OUTPUT EXPLANATION
// ============================================================


export type AnalysisOutputRecommendationStatus =
  | "recommended_terminal"
  | "superseded_intermediate";


export type AnalysisOutputRecommendationFacts = {
  workflow_id:
    string;

  dataset_id:
    string;

  dataset_filename:
    string;

  stage:
    string;

  rows:
    number;

  columns:
    number;

  recommendation_status:
    AnalysisOutputRecommendationStatus;

  is_terminal:
    boolean;

  direct_parent_dataset_ids:
    string[];

  ancestor_dataset_ids:
    string[];

  ancestor_dataset_filenames:
    string[];

  root_dataset_ids:
    string[];

  root_dataset_filenames:
    string[];

  lineage_depth:
    number;

  replaced_dataset_ids:
    string[];

  evidence_refs:
    string[];

  deterministic_reasons:
    string[];

  rule_version:
    string;
};


export type AnalysisOutputExplanation = {
  workflow_id:
    string;

  dataset_id:
    string;

  dataset_filename:
    string;

  recommendation_status:
    AnalysisOutputRecommendationStatus;

  confidence:
    number;

  title:
    string;

  explanation:
    string;

  user_message:
    string;

  referenced_dataset_ids:
    string[];

  cautions:
    string[];

  python_validated:
    boolean;

  executable:
    boolean;

  model:
    string;

  rule_version:
    string;
};


export type PreparationOutputExplanationResponse = {
  workflow_id:
    string;

  dataset_id:
    string;

  dataset_filename:
    string;

  facts:
    AnalysisOutputRecommendationFacts;

  explanation:
    AnalysisOutputExplanation |
    null;

  ai_error:
    string |
    null;

  recommended:
    boolean;

  api_version:
    string;

  rule_version:
    string;
};


// ============================================================
// CONTROLLED COMBINE WORKFLOW
// ============================================================


export type PreparationCombineJoinKey = {
  left_column:
    string;

  right_column:
    string;
};


export type PreparationCombineIntent = {
  request_id:
    string;

  left_dataset_id:
    string;

  left_dataset_filename:
    string;

  right_dataset_id:
    string;

  right_dataset_filename:
    string;

  join_type:
    string;

  keys:
    PreparationCombineJoinKey[];

  expected_cardinality:
    string;

  output_dataset_id:
    string;

  output_dataset_filename:
    string;

  left_suffix:
    string;

  right_suffix:
    string;
};


export type PreparationCombinePlan = {
  ready_for_approval?:
    boolean;

  rule_version?:
    string;

  joins?:
    Record<
      string,
      unknown
    >[];

  [
    key:
      string
  ]:
    unknown;
};


export type PreparationCombineDiscoveryView = {
  workflow_id:
    string;

  active_dataset_ids:
    string[];

  reason:
    string;

  has_candidate:
    boolean;

  ready_for_approval:
    boolean;

  intent:
    PreparationCombineIntent |
    null;

  plan:
    PreparationCombinePlan |
    null;

  service_version:
    string;
};


export type PreparationCombineDiscoveryResponse = {
  discovery:
    PreparationCombineDiscoveryView;

  session:
    PreparationSessionView;

  api_version:
    string;
};


export type PreparationCombineExecutionResponse = {
  workflow_id:
    string;

  request_id:
    string;

  output_dataset_id:
    string;

  output_dataset_filename:
    string;

  rows:
    number;

  columns:
    number;

  parent_dataset_ids:
    string[];

  validation:
    Record<
      string,
      unknown
    >;

  next_discovery:
    PreparationCombineDiscoveryView;

  session:
    PreparationSessionView;

  service_version:
    string;

  api_version:
    string;
};


export type PreparationIssueSeverity =
  | "important"
  | "moderate"
  | "minor";


export type QualityIssueEvidenceView = {
  observed_count:
    number;

  affected_ratio:
    number;

  examples:
    string[];

  details:
    Record<
      string,
      unknown
    >;
};


export type CleaningProposalView = {
  operation:
    string;

  automatic_safe:
    boolean;

  description:
    string;

  requires_user_confirmation:
    boolean;

  parameters:
    Record<
      string,
      unknown
    >;
};


export type DataQualityIssueView = {
  issue_id:
    string;

  dataset_id:
    string;

  dataset_filename:
    string;

  column:
    string |
    null;

  kind:
    string;

  severity:
    PreparationIssueSeverity;

  title:
    string;

  explanation:
    string;

  evidence:
    QualityIssueEvidenceView;

  proposal:
    CleaningProposalView;

  semantic_review_recommended:
    boolean;
};


export type DatasetQualitySummaryView = {
  dataset_id:
    string;

  dataset_filename:
    string;

  row_count:
    number;

  column_count:
    number;

  missing_cell_count:
    number;

  missing_cell_ratio:
    number;

  duplicate_row_count:
    number;

  issue_count:
    number;

  important_count:
    number;

  moderate_count:
    number;

  minor_count:
    number;
};


export type DataQualityReportView = {
  status:
    string;

  dataset_count:
    number;

  total_rows:
    number;

  total_columns:
    number;

  issue_count:
    number;

  important_count:
    number;

  moderate_count:
    number;

  minor_count:
    number;

  semantic_review_count:
    number;

  datasets:
    DatasetQualitySummaryView[];

  issues:
    DataQualityIssueView[];

  notes:
    string[];

  rule_version:
    string;
};


export type CleaningActionView = {
  action_id:
    string;

  dataset_id:
    string;

  dataset_filename:
    string;

  kind:
    string;

  column:
    string |
    null;

  title:
    string;

  rationale:
    string;

  safe_candidate:
    boolean;

  requires_user_confirmation:
    boolean;

  affected_rows_estimate:
    number;

  before_examples:
    string[];

  after_examples:
    string[];

  parameters:
    Record<
      string,
      unknown
    >;
};


export type CleaningPlanView = {
  status:
    string;

  dataset_count:
    number;

  action_count:
    number;

  safe_candidate_count:
    number;

  confirmation_required_count:
    number;

  protected_issue_count:
    number;

  actions:
    CleaningActionView[];

  notes:
    string[];

  rule_version:
    string;
};


export type CleaningActionResultView = {
  action_id:
    string;

  status:
    string;

  affected_rows_actual:
    number;

  rows_before:
    number;

  rows_after:
    number;

  details:
    Record<
      string,
      unknown
    >;
};


export type DatasetCleaningProvenanceView = {
  dataset_id:
    string;

  dataset_filename:
    string;

  rows_before:
    number;

  rows_after:
    number;

  columns_before:
    number;

  columns_after:
    number;

  source_fingerprint:
    string;

  derived_fingerprint:
    string;

  applied_action_ids:
    string[];

  skipped_action_ids:
    string[];
};


export type CleaningExecutionView = {
  status:
    string;

  dataset_count:
    number;

  applied_action_count:
    number;

  skipped_action_count:
    number;

  blocked_action_count:
    number;

  action_results:
    CleaningActionResultView[];

  provenance:
    DatasetCleaningProvenanceView[];

  notes:
    string[];

  rule_version:
    string;
};


export type CleaningApplyResponseView = {
  status:
    string;

  quality_report:
    DataQualityReportView;

  cleaning_plan:
    CleaningPlanView;

  execution:
    CleaningExecutionView;

  derived_datasets:
    Array<{
      dataset_id:
        string;

      dataset_filename:
        string;

      rows_before:
        number;

      rows_after:
        number;

      columns_before:
        number;

      columns_after:
        number;

      preview_rows:
        Array<
          Record<
            string,
            unknown
          >
        >;
    }>;

  notes:
    string[];
};



export type SemanticVerdictView =
  | "merge_values"
  | "keep_separate"
  | "flag_for_review"
  | "contextualize"
  | "no_change"
  | "abstain";


export type SemanticDecisionView = {
  issue_id:
    string;

  dataset_id:
    string;

  dataset_filename:
    string;

  column:
    string |
    null;

  kind:
    string;

  verdict:
    SemanticVerdictView;

  confidence:
    number;

  rationale:
    string;

  source_values:
    string[];

  canonical_value:
    string |
    null;

  user_message:
    string;

  python_validated:
    boolean;

  executable:
    boolean;

  requires_user_confirmation:
    boolean;

  validation_notes:
    string[];
};


export type SemanticReviewReportView = {
  status:
    string;

  model:
    string;

  candidate_count:
    number;

  decision_count:
    number;

  merge_proposal_count:
    number;

  abstention_count:
    number;

  decisions:
    SemanticDecisionView[];

  notes:
    string[];

  rule_version:
    string;
};


export type SemanticCleaningActionView = {
  action_id:
    string;

  issue_id:
    string;

  dataset_id:
    string;

  dataset_filename:
    string;

  column:
    string;

  source_values:
    string[];

  suggested_canonical_value:
    string;

  allowed_canonical_values:
    string[];

  confidence:
    number;

  rationale:
    string;

  requires_user_confirmation:
    boolean;

  python_validated:
    boolean;
};


export type SemanticCleaningPlanView = {
  status:
    string;

  action_count:
    number;

  actions:
    SemanticCleaningActionView[];

  notes:
    string[];

  rule_version:
    string;
};


export type SemanticCleaningChoiceView = {
  action_id:
    string;

  canonical_value:
    string;
};


export type SemanticCleaningActionResultView = {
  action_id:
    string;

  status:
    "applied" |
    "skipped";

  dataset_id:
    string;

  column:
    string;

  source_values:
    string[];

  canonical_value:
    string |
    null;

  affected_rows_actual:
    number;

  details:
    Record<
      string,
      unknown
    >;
};


export type SemanticDatasetProvenanceView = {
  dataset_id:
    string;

  dataset_filename:
    string;

  rows_before:
    number;

  rows_after:
    number;

  source_fingerprint:
    string;

  derived_fingerprint:
    string;

  applied_action_ids:
    string[];

  changed_cell_count:
    number;
};


export type SemanticCleaningExecutionView = {
  status:
    string;

  dataset_count:
    number;

  applied_action_count:
    number;

  skipped_action_count:
    number;

  changed_cell_count:
    number;

  action_results:
    SemanticCleaningActionResultView[];

  provenance:
    SemanticDatasetProvenanceView[];

  notes:
    string[];

  rule_version:
    string;
};


export type SemanticCleaningApplyResponseView = {
  plan:
    SemanticCleaningPlanView;

  execution:
    SemanticCleaningExecutionView;
};

export type SemanticConfirmationReportView = {
  confirmed: boolean;

  decision_count: number;
  confirmed_issue_count: number;
  manual_resolution_count: number;

  merge_action_count: number;
  applied_merge_action_count: number;
  skipped_merge_action_count: number;

  confirmed_issue_ids: string[];
  manually_resolved_issue_ids: string[];
  unresolved_issue_ids: string[];
  unresolved_reasons: string[];

  notes: string[];
  rule_version: string;
};

export type PreparationUiStateView = {
  workflow_id:
    string;

  revision:
    number;

  quality_report:
    DataQualityReportView |
    null;

  cleaning_plan:
    CleaningPlanView |
    null;

  cleaning_execution:
    CleaningExecutionView |
    null;

  semantic_review:
    SemanticReviewReportView |
    null;

  semantic_cleaning_plan:
    SemanticCleaningPlanView |
    null;

  semantic_cleaning_execution:
    SemanticCleaningExecutionView |
    null;

  semantic_confirmation:
    SemanticConfirmationReportView |
    null;

  applied_semantic_choices:
    SemanticCleaningChoiceView[];

  confirmed_semantic_issue_ids:
    string[];

  semantic_manual_resolutions:
    Array<{
      issue_id:
        string;

      note:
        string;
    }>;

  storage:
    string;

  persistent:
    boolean;

  rule_version:
    string;
};

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


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

  selected_analysis_dataset_ids:
    string[];

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

  selected_analysis_dataset_ids:
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

  notes:
    string[];
};


export type CreatePreparationSessionRequest = {
  selected_analysis_dataset_ids:
    string[];
};
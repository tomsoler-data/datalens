import type {
  ReportChartDatum,
  ReportRequestedFinding,
} from "../../app/types";


export type AIPlannerBindingView = {
  role:
    string;

  column:
    string;

  dataset_id:
    string |
    null;

  dataset_filename:
    string |
    null;

  semantic_concept:
    string |
    null;

  analysis_kind:
    string |
    null;
};


export type AIPlannerContractView = {
  contract_id:
    string;

  contract_version:
    string;

  origin:
    string;

  status:
    string;

  title:
    string;

  request_text:
    string;

  family:
    string;

  required_dataset_ids:
    string[];

  required_dataset_filenames:
    string[];

  analytical_grain:
    string |
    null;

  bindings:
    AIPlannerBindingView[];

  reasons:
    string[];

  blockers:
    string[];

  planner_confidence:
    number |
    null;
};


export type AIPlannerProposalView = {
  decision:
    string;

  title:
    string;

  family:
    string;

  dataset_id:
    string |
    null;

  analytical_grain:
    string |
    null;

  x_column:
    string |
    null;

  y_column:
    string |
    null;

  group_column:
    string |
    null;

  value_column:
    string |
    null;

  time_column:
    string |
    null;

  dimension_column:
    string |
    null;

  entity_column:
    string |
    null;

  aggregation_function:
    string;

  ranking_order:
    string;

  ranking_limit:
    number |
    null;

  window_operation:
    string;

  window_size:
    number |
    null;

  blockers:
    string[];

  reasons:
    string[];

  confidence:
    number;
};


export type AIPlannerItemView = {
  proposal_index:
    number;

  validation_status:
    "validated" |
    "blocked" |
    "ambiguous" |
    "rejected";

  raw_proposal?:
    AIPlannerProposalView |
    null;

  proposal:
    AIPlannerProposalView;

  contract:
    AIPlannerContractView |
    null;

  errors:
    string[];

  warnings:
    string[];

  normalizations?:
    string[];
};


export type AIPlannerReportView = {
  status:
    string;

  objective:
    string;

  model:
    string;

  proposal_count:
    number;

  validated_count:
    number;

  blocked_count:
    number;

  ambiguous_count:
    number;

  rejected_count:
    number;

  items:
    AIPlannerItemView[];

  attempt_count?:
    number;

  retry_count?:
    number;

  retry_triggered?:
    boolean;

  retry_feedback?:
    string[];

  normalization_count?:
    number;

  normalization_applied?:
    boolean;

  planner_rule_version:
    string;
};


export type AINativeAttemptView = {
  attempt_index:
    number;

  prompt_variant:
    "standard" |
    "mandatory_retry";

  tool_call_count:
    number;

  assistant_content:
    string;

  selected_tool_name:
    string |
    null;

  errors:
    string[];
};


export type AINativeExecutionResultView = {
  analysis_id?:
    string;

  dataset_id?:
    string;

  dataset_filename?:
    string;

  title:
    string;

  family:
    string;

  execution_status:
    string;

  chart_type:
    string;

  summary:
    string[];

  metrics:
    Record<
      string,
      unknown
    >;

  chart_data?:
    ReportChartDatum[];

  statistical_decision?:
    Record<
      string,
      unknown
    > |
    null;

  statistical_result?:
    Record<
      string,
      unknown
    > |
    null;

  warnings:
    string[];

  limitations:
    string[];

  execution_rule_version?:
    string;
};


export type AINativeExecutionTraceView = {
  tool_name:
    string |
    null;

  execution_status:
    string;

  dataset_id:
    string |
    null;

  dataset_filename:
    string |
    null;

  arguments:
    {
      family?:
        string;

      dataset_ids?:
        string[];

      analytical_grain?:
        string |
        null;

      variables?:
        Record<
          string,
          string
        >;
    };

  result:
    AINativeExecutionResultView |
    null;

  errors:
    string[];

  warnings:
    string[];
};


export type AINativeToolView = {
  model:
    string;

  contract_family?:
    string;

  available_tools?:
    string[];

  expected_tool?:
    string |
    null;

  tool_call_received:
    boolean;

  requested_tool:
    string |
    null;

  requested_arguments:
    Record<
      string,
      unknown
    >;

  validation_status:
    "validated" |
    "rejected";

  validation_errors:
    string[];

  attempt_count:
    number;

  retry_count:
    number;

  attempts:
    AINativeAttemptView[];

  execution:
    AINativeExecutionTraceView |
    null;

  native_tool_rule_version:
    string;
};


export type AINativePipelineItemView = {
  contract_id:
    string;

  family:
    string;

  pipeline_status:
    "executed" |
    "not_supported" |
    "rejected";

  native_tool:
    AINativeToolView |
    null;

  errors:
    string[];

  warnings:
    string[];
};


export type AINativePipelineReportView = {
  trace_id?:
    string |
    null;

  analysis_id?:
    string |
    null;

  analysis_source_type?:
    "initial_request" |
    "follow_up_prompt" |
    "document_request" |
    "automatic" |
    null;

  requested_finding?:
    ReportRequestedFinding |
    null;

  status:
    string;

  planner:
    AIPlannerReportView;

  planner_model:
    string;

  tool_model:
    string;

  supported_native_families?:
    string[];

  validated_contract_count:
    number;

  pipeline_item_count:
    number;

  executed_count:
    number;

  not_supported_count:
    number;

  rejected_count:
    number;

  items:
    AINativePipelineItemView[];

  notes:
    string[];

  pipeline_rule_version:
    string;
};


export type ReportSelectionItemView = {
  analysis_id:
    string;

  source_type:
    "initial_request" |
    "follow_up_prompt" |
    "document_request" |
    "automatic";

  objective:
    string;

  trace_id:
    string;

  report_order:
    number;

  added_at_utc:
    string;

  executed:
    boolean;
};


export type ReportAvailableAnalysisDetailView = {
  analysis_id:
    string;

  workflow_id:
    string;

  trace_id:
    string;

  source_type:
    ReportSelectionItemView[
      "source_type"
    ];

  objective:
    string;

  executed:
    boolean;

  executed_count:
    number;

  pipeline_payload:
    AINativePipelineReportView;

  created_at_utc:
    string;
};

export type ReportPromptAnalysisView = {
  id:
    string;

  source_type:
    "initial_request" |
    "follow_up_prompt" |
    "document_request" |
    "automatic";

  source_label:
    string;

  objective:
    string;

  report:
    AINativePipelineReportView;
};

export type AnalysisFollowUpTurn = {
  id:
    string;

  objective:
    string;

  report:
    AINativePipelineReportView;

  included_in_report:
    boolean;
};

type DocumentCitationView = {
  chunk_id:
    string;

  document_id:
    string;

  filename:
    string;

  source_locator:
    string;

  page_number:
    number |
    null;
};

type DocumentClaimView = {
  category:
    string;

  statement:
    string;

  evidence_quote:
    string;

  evidence_unit_id:
    number;

  context_quote:
    string |
    null;

  context_evidence_unit_id:
    number |
    null;

  citation:
    DocumentCitationView;
};

type DocumentSummaryItemView = {
  document_id:
    string;

  filename:
    string;

  summary_points:
    DocumentClaimView[];

  analytical_requests:
    DocumentClaimView[];

  verified_claim_count:
    number;

  source_chunk_count:
    number;
};

export type DocumentSummaryView = {
  status:
    string;

  document_count:
    number;

  chunk_count:
    number;

  verified_claim_count:
    number;

  summary_point_count:
    number;

  analytical_request_count:
    number;

  summary_points:
    DocumentClaimView[];

  analytical_requests:
    DocumentClaimView[];

  documents:
    DocumentSummaryItemView[];

  warnings:
    string[];

  abstention_reason:
    string |
    null;

  model:
    string;
};

export type RequestedPlanItemView = {
  request_id:
    string;

  request_text:
    string;

  source_filename:
    string;

  source_locator:
    string;

  page_number:
    number |
    null;

  kind:
    string;

  status:
    "ready" |
    "blocked" |
    "ambiguous";

  blockers:
    string[];
};

export type RequestedPlanView = {
  request_count:
    number;

  ready_count:
    number;

  blocked_count:
    number;

  ambiguous_count:
    number;

  requests:
    RequestedPlanItemView[];
};


export type RequestedTimeGranularity =
  | "day"
  | "week"
  | "month"
  | "quarter"
  | "year";


export type EntityOutlierFindingEvidenceView = {
  metric:
    string;

  metric_label:
    string;

  family:
    string;

  family_label:
    string;

  value:
    number;

  direction:
    string;

  distance_iqr:
    number;
};


export type EntityOutlierFindingProfileView = {
  entity:
    string;

  severity:
    string;

  dominant_family:
    string;

  dominant_family_label:
    string;

  signal_count:
    number;

  max_distance_iqr:
    number;

  title:
    string;

  explanation:
    string;

  evidence:
    EntityOutlierFindingEvidenceView[];
};


export type EntityOutlierFindingView = {
  analysis_id:
    string;

  status:
    "ready" |
    "blocked";

  title:
    string;

  family:
    "entity_outlier";

  kind:
    "customer_entity_outlier_detection";

  dataset_id:
    string |
    null;

  dataset_filename:
    string |
    null;

  entity_column:
    string |
    null;

  entity_count:
    number;

  raw_flagged_entity_count:
    number;

  priority_profile_count:
    number;

  behavioral_signal_count:
    number;

  summary:
    string[];

  priority_profiles:
    EntityOutlierFindingProfileView[];

  caveats:
    string[];

  methodology:
    string[];

  blockers:
    string[];

  adapter_rule_version:
    string;
};

export type ReportSelectionDetailView = {
  selection:
    ReportSelectionItemView;

  pipeline_payload:
    AINativePipelineReportView;
};

export type ReportSelectionDetailsView = {
  workflow_id:
    string;

  revision:
    number;

  selected_count:
    number;

  analyses:
    ReportSelectionDetailView[];

  rule_version:
    string;
};

export type ReportAvailableAnalysisListView = {
  workflow_id:
    string;

  count:
    number;

  analyses:
    ReportAvailableAnalysisDetailView[];

  rule_version:
    string;
};

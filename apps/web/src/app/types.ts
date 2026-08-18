export type AnalysisStatus =
  | "complete"
  | "needs_information"
  | "not_applicable";


export type AnalysisGoal =
  | "linear_association"
  | "monotonic_association"
  | "general_association";


export type AnalysisMode =
  | "confirmatory"
  | "exploratory";


export type VariableKind =
  | "continuous"
  | "ordinal"
  | "binary"
  | "nominal"
  | "temporal"
  | "unknown";


export type ColumnAnalysisKind =
  | "quantitative"
  | "temporal"
  | "categorical"
  | "boolean"
  | "unknown";


export type DatasetColumnManifest = {
  name:
    string;

  dtype:
    string;

  missing_count:
    number;

  missing_ratio:
    number;

  unique_count:
    number;

  unique_ratio:
    number;

  unique_candidate:
    boolean;

  analysis_kind:
    ColumnAnalysisKind;

  correlation_eligible:
    boolean;

  analysis_note:
    string;
};


export type CorrelationCompatibility = {
  status:
    | "ready"
    | "not_available";

  candidate_columns:
    string[];

  default_x_column:
    string | null;

  default_y_column:
    string | null;

  reasons:
    string[];
};


export type DatasetManifest = {
  dataset_id:
    string;

  filename:
    string;

  extension:
    string;

  row_count:
    number;

  column_count:
    number;

  memory_bytes:
    number;

  columns:
    DatasetColumnManifest[];

  correlation_compatibility:
    CorrelationCompatibility;

  warnings:
    string[];
};


export type MultiDatasetIngestion = {
  status:
    "ready";

  dataset_count:
    number;

  total_rows:
    number;

  datasets:
    DatasetManifest[];

  warnings:
    string[];

  ingestion_rule_version:
    string;
};


/* ============================================================
   LEGACY CORRELATION CONTRACTS
============================================================ */

export type DashboardKpi = {
  key:
    string;

  label:
    string;

  kind:
    string;

  value:
    string;

  source_reference:
    string;

  source_field:
    string;
};


export type DashboardChart = {
  visualization_reference:
    string;

  chart_type:
    string;

  purpose:
    | "relationship"
    | "diagnostic";

  x_column:
    string;

  y_column:
    string;

  aggregation:
    string;

  trend:
    string;

  show_raw_points:
    boolean;

  show_missing_summary:
    boolean;

  reasons:
    string[];
};


export type DashboardStatisticalResult = {
  statistic_reference:
    string;

  test:
    string;

  relationship_type:
    string;

  coefficient_name:
    string;

  coefficient:
    string;

  p_value:
    string;

  alpha:
    string;

  statistically_significant:
    boolean;

  n:
    number;

  inference_method:
    string;

  permutation_mode:
    string | null;
};


export type DashboardDecision = {
  decision_reference:
    string;

  status:
    string;

  analysis_goal:
    AnalysisGoal;

  analysis_mode:
    AnalysisMode;

  selected_test:
    string | null;

  selection_is_data_driven:
    boolean;

  reasons:
    string[];

  missing_information:
    string[];
};


export type DashboardSpec = {
  dashboard_id:
    string;

  status:
    string;

  title:
    string;

  subtitle:
    string;

  summary:
    string;

  x_column:
    string;

  y_column:
    string;

  kpis:
    DashboardKpi[];

  chart:
    DashboardChart | null;

  statistical_result:
    DashboardStatisticalResult | null;

  decision:
    DashboardDecision;

  action_required:
    string[];

  warnings:
    string[];

  evidence: {
    decision:
      string;

    statistic:
      string | null;

    visualization:
      string;
  };

  dashboard_rule_version:
    string;
};


export type CorrelationDecision = {
  status:
    string;

  analysis_goal:
    AnalysisGoal;

  analysis_mode:
    AnalysisMode;

  x_column:
    string;

  y_column:
    string;

  selected_test:
    string | null;

  reasons:
    string[];

  missing_information:
    string[];

  warnings:
    string[];
};


export type VisualizationDecision = {
  visualization_id:
    string;

  status:
    string;

  purpose:
    | "relationship"
    | "diagnostic";

  chart_type:
    string | null;

  x_column:
    string;

  y_column:
    string;

  aggregation:
    string;

  trend:
    string;

  show_raw_points:
    boolean;

  show_missing_summary:
    boolean;

  selection_is_data_driven:
    boolean;

  reasons:
    string[];

  warnings:
    string[];

  compatible_alternatives:
    string[];
};


export type EvidenceRecord = {
  evidence_id:
    string;

  source_type:
    string;

  dataset:
    string;

  producer:
    string;

  rule_version:
    string;

  depends_on:
    string[];

  data:
    Record<
      string,
      unknown
    >;
};


export type AnalysisExecution = {
  selected_test:
    string;

  inference_method_used:
    string;

  permutation_mode:
    string | null;

  result: {
    test:
      string;

    relationship_type:
      string;

    coefficient_name:
      string;

    coefficient:
      number;

    p_value:
      number;

    n:
      number;

    alpha:
      number;

    statistically_significant:
      boolean;
  };
};


export type AnalysisRun = {
  analysis_id:
    string;

  dataset:
    string;

  status:
    AnalysisStatus;

  decision:
    CorrelationDecision;

  execution:
    AnalysisExecution | null;

  visualization:
    VisualizationDecision;

  dashboard:
    DashboardSpec;

  evidence: {
    dataset:
      string;

    evidence:
      EvidenceRecord[];

    evidence_rule_version:
      string;
  };

  pipeline_rule_version:
    string;
};


/* ============================================================
   REQUEST PLANNER
============================================================ */

export type RequestPlanningStatus =
  | "ready"
  | "blocked"
  | "ambiguous";


export type RequestedAnalysisKind =
  | "revenue_moving_average"
  | "revenue_by_category"
  | "customers_by_period"
  | "transaction_count"
  | "products_sold_count"
  | "top_products"
  | "flop_products"
  | "product_category_distribution"
  | "b2b_revenue_distribution"
  | "lorenz_curve"
  | "gender_category_association"
  | "age_total_amount_association"
  | "age_frequency_association"
  | "age_average_basket_association"
  | "age_category_association"
  | "unknown";


export type RequestedColumnMatch = {
  concept:
    string;

  dataset_id:
    string;

  dataset_filename:
    string;

  column:
    string;

  analysis_kind:
    string;

  match_score:
    number;

  reasons:
    string[];
};


export type RequestedAnalysisPlan = {
  request_id:
    string;

  request_text:
    string;

  context_text:
    string | null;

  evidence_quote:
    string;

  source_filename:
    string;

  source_locator:
    string;

  page_number:
    number | null;

  source_chunk_id:
    string;

  evidence_unit_id:
    number;

  kind:
    RequestedAnalysisKind;

  status:
    RequestPlanningStatus;

  target_family:
    string | null;

  matched_columns:
    RequestedColumnMatch[];

  required_dataset_ids:
    string[];

  required_dataset_filenames:
    string[];

  required_operations:
    string[];

  reasons:
    string[];

  blockers:
    string[];
};


export type RequestedAnalysisPlanReport = {
  status:
    "ready";

  request_count:
    number;

  ready_count:
    number;

  blocked_count:
    number;

  ambiguous_count:
    number;

  requests:
    RequestedAnalysisPlan[];

  planner_notes:
    string[];

  planner_rule_version:
    string;
};


/* ============================================================
   REQUESTED ANALYSIS EXECUTION
============================================================ */

export type RequestedExecutionStatus =
  | "complete"
  | "descriptive_only"
  | "needs_information"
  | "needs_specialized_method"
  | "skipped"
  | "failed"
  | "not_executed"
  | "not_supported_yet";


export type RequestedInferentialStatus =
  | "executed"
  | "not_selected"
  | "not_applicable";


export type RequestedStatisticalMode =
  | "exploratory"
  | "confirmatory";


export type RequestedAnalysisExecution = {
  request_id:
    string;

  request_text:
    string;

  kind:
    RequestedAnalysisKind;

  plan_status:
    RequestPlanningStatus;

  execution_status:
    RequestedExecutionStatus;

  inferential_status:
    RequestedInferentialStatus | null;

  source_filename:
    string;

  source_locator:
    string;

  evidence_quote:
    string;

  dataset_id:
    string | null;

  dataset_filename:
    string | null;

  analytical_grain:
    string | null;

  analysis_mode:
    RequestedStatisticalMode | null;

  variables:
    Record<
      string,
      string
    >;

  descriptive_statistics:
    Record<
      string,
      unknown
    >;

  result:
    Record<
      string,
      unknown
    > | null;

  warnings:
    string[];

  limitations:
    string[];

  executor_rule_version:
    string;
};


export type RequestedAnalysisExecutionReport = {
  status:
    "ready";

  request_count:
    number;

  attempted_count:
    number;

  complete_count:
    number;

  descriptive_only_count:
    number;

  needs_information_count:
    number;

  needs_specialized_method_count:
    number;

  skipped_count:
    number;

  failed_count:
    number;

  not_executed_count:
    number;

  not_supported_yet_count:
    number;

  inference_executed_count:
    number;

  inference_abstained_count:
    number;

  results:
    RequestedAnalysisExecution[];

  executor_notes:
    string[];

  executor_rule_version:
    string;
};


/* ============================================================
   DOCUMENT SUMMARY

   The exact document-summary UI contract will be narrowed
   when we start rendering this section. For now, preserve the
   complete backend object without inventing frontend fields.
============================================================ */

export type DocumentSummaryReport = {
  status?:
    string;

  [key: string]:
    unknown;
};


/* ============================================================
   UNIFIED MULTI-DATASET REPORT
============================================================ */

export type UnifiedReportInventory = {
  dataset_count:
    number;

  discovered_analysis_count:
    number;

  executed_analysis_count:
    number;

  requested_finding_count:
    number;

  main_finding_count:
    number;

  additional_finding_count:
    number;

  diagnostic_count:
    number;

  quality_check_count:
    number;

  context_analysis_count:
    number;

  blocked_analysis_count:
    number;
};


export type ReportDatasetSummary = {
  dataset_id:
    string;

  filename:
    string;

  row_count:
    number;

  column_count:
    number;

  columns:
    string[];
};


export type ReportChartDatum =
  Record<
    string,
    unknown
  >;


/* ============================================================
   EXPLORATORY REPORT FINDING
============================================================ */

export type ReportFinding = {
  analysis_id?:
    string;

  title:
    string;

  family:
    string;

  datasets:
    string[];

  summary:
    string[];

  metrics:
    Record<
      string,
      unknown
    >;

  chart_type:
    string | null;

  chart_data:
    ReportChartDatum[];

  role?:
    string;

  scope?:
    string;

  tier?:
    string;

  execution_status?:
    string;

  interestingness_score?:
    number;

  signal_score?:
    number;

  coverage_score?:
    number;

  consistency_score?:
    number;

  status?:
    string;

  strength?:
    string | null;

  direction?:
    string | null;

  sample_size?:
    number | null;

  period_count?:
    number | null;

  readiness?:
    string;

  caveats?:
    string[];

  limitations?:
    string[];

  reasons?:
    string[];
};


/* ============================================================
   REQUESTED REPORT FINDING
============================================================ */

export type ReportRequestedFinding = {
  request_id:
    string;

  analysis_id:
    string;

  title:
    string;

  origin:
    "requested";

  kind:
    RequestedAnalysisKind | string;

  scope:
    string;

  family:
    string;

  execution_status:
    RequestedExecutionStatus | string;

  inferential_status:
    RequestedInferentialStatus | string | null;

  analysis_mode:
    RequestedStatisticalMode | string | null;

  dataset_id:
    string | null;

  datasets:
    string[];

  analytical_grain:
    string | null;

  variables:
    Record<
      string,
      string
    >;

  sample_size:
    number;

  summary:
    string[];

  reasons:
    string[];

  caveats:
    string[];

  chart_type:
    string | null;

  chart_data:
    ReportChartDatum[];

  metrics:
    Record<
      string,
      unknown
    >;

  source_filename:
    string;

  source_locator:
    string;

  page_number:
    number | null;

  source_chunk_id:
    string | null;

  evidence_unit_id:
    number | null;

  evidence_quote:
    string;

  adapter_rule_version:
    string;
};


export type ReportQualityItem = {
  analysis_id:
    string;

  dataset:
    string;

  row_count:
    number;

  column_count:
    number;

  missing_cells:
    number;

  missing_ratio:
    number;

  duplicate_rows:
    number;

  duplicate_ratio:
    number;

  completely_missing_columns:
    string[];

  constant_columns:
    string[];

  summary:
    string[];
};


export type ReportBlockedAnalysis = {
  analysis_id:
    string;

  title:
    string;

  family:
    string;

  datasets:
    string[];

  reason:
    string;

  caveats:
    string[];

  discovery_priority_score:
    number;
};


export type UnifiedAnalysisReport = {
  status:
    string;

  title:
    string;

  datasets:
    ReportDatasetSummary[];

  inventory:
    UnifiedReportInventory;

  executive_summary:
    string[];

  requested_findings:
    ReportRequestedFinding[];

  main_findings:
    ReportFinding[];

  additional_findings:
    ReportFinding[];

  diagnostics:
    ReportFinding[];

  quality:
    ReportQualityItem[];

  context_analyses:
    ReportFinding[];

  blocked_analyses:
    ReportBlockedAnalysis[];

  methodology_notes:
    string[];

  report_rule_version:
    string;
};


/* ============================================================
   RAG — SHARED TYPES
============================================================ */

export type RagRelationType =
  | "explicit_request"
  | "business_rule"
  | "business_definition"
  | "objective_support"
  | "interpretation_context"
  | "methodological_context"
  | "not_relevant";


export type RagStrength =
  | "direct"
  | "supporting"
  | "none";


export type RagSearchHit = {
  rank:
    number;

  score:
    number;

  chunk_id:
    string;

  document_id:
    string;

  filename:
    string;

  extension:
    string;

  chunk_index:
    number;

  page_number:
    number | null;

  source_locator:
    string;

  text:
    string;

  character_count:
    number;
};


export type RagCitation = {
  chunk_id:
    string;

  filename:
    string;

  source_locator:
    string;

  page_number:
    number | null;
};


export type RagHitRelevanceDecision = {
  rank:
    number;

  chunk_id:
    string;

  filename:
    string;

  source_locator:
    string;

  score:
    number;

  verdict:
    | "relevant"
    | "not_relevant";

  relation_type:
    RagRelationType;

  strength:
    RagStrength;

  reason:
    string;
};


/* ============================================================
   RAG — ANALYTICAL CONTRACT
============================================================ */

export type AnalyticalContract = {
  family:
    string;

  title:
    string;

  measure_column:
    string | null;

  group_column:
    string | null;

  x_column:
    string | null;

  y_column:
    string | null;

  time_column:
    string | null;

  measure_semantics:
    string | null;

  analytical_relationship:
    string;
};


/* ============================================================
   RAG — DETERMINISTIC DOCUMENT CONTEXT
============================================================ */

export type DeterministicDocumentContext = {
  status:
    | "available"
    | "abstained";

  relation_type:
    RagRelationType | null;

  strength:
    RagStrength | null;

  message:
    string;

  citation:
    RagCitation | null;
};


/* ============================================================
   RAG — GROUNDED EXPLANATION
============================================================ */

export type VerifiedGroundedClaim = {
  statement:
    string;

  evidence_quote:
    string;

  citation:
    RagCitation;
};


export type VerifiedRagExplanation = {
  status:
    | "ready"
    | "abstained";

  explanation:
    string;

  claims:
    VerifiedGroundedClaim[];

  abstention_reason:
    string | null;

  explanation_rule_version:
    string;

  model:
    string;
};


/* ============================================================
   RAG — FINDING CONTEXT
============================================================ */

export type FindingRagContext = {
  analysis_id:
    string;

  title:
    string;

  family:
    string;

  analytical_contract:
    AnalyticalContract;

  query:
    string;

  relevance_finding_text:
    string;

  hits:
    RagSearchHit[];

  relevance_decisions:
    RagHitRelevanceDecision[];

  accepted_hits:
    RagSearchHit[];

  documentary_context:
    DeterministicDocumentContext;

  abstained:
    boolean;

  abstention_reason:
    string | null;

  explanation:
    VerifiedRagExplanation;

  explanation_error:
    string | null;
};


/* ============================================================
   RAG — REPORT
============================================================ */

export type RagContextReport = {
  status:
    "ready";

  objective:
    string | null;

  document_count:
    number;

  chunk_count:
    number;

  finding_count:
    number;

  top_k:
    number;

  model:
    string;

  relevance_model:
    string;

  explanation_model:
    string;

  validated_candidate_count:
    number;

  accepted_hit_count:
    number;

  accepted_finding_count:
    number;

  abstained_finding_count:
    number;

  documentary_context_available_count:
    number;

  explanation_ready_count:
    number;

  explanation_abstained_count:
    number;

  explanation_error_count:
    number;

  contexts:
    FindingRagContext[];

  retrieval_rule_version:
    string;

  relevance_rule_version:
    string;

  explanation_rule_version:
    string;

  context_rule_version:
    string;
};


/* ============================================================
   CONTEXTUALIZED ANALYSIS RESPONSE
============================================================ */

export type ContextualizedAnalysisResponse = {
  analysis:
    UnifiedAnalysisReport;

  document_summary:
    DocumentSummaryReport;

  requested_analysis_plan:
    RequestedAnalysisPlanReport;

  requested_analysis_execution:
    RequestedAnalysisExecutionReport;

  rag:
    RagContextReport;
};
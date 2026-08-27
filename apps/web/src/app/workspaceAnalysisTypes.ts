import type {
  EntityOutlierFindingView,
} from "../components/analysis/analysisTypes";

import type {
  PrioritizationAuditView,
} from "./PrioritizationAuditPanel";

import type {
  ContextualizedAnalysisResponse,
  UnifiedAnalysisReport,
} from "./types";


export type RoutedUnifiedAnalysisReportView =
  UnifiedAnalysisReport & {
    entity_outlier_finding?:
      EntityOutlierFindingView |
      null;

    prioritization_audit?:
      PrioritizationAuditView |
      null;
  };


export type RoutedContextualizedAnalysisResponseView =
  Omit<
    ContextualizedAnalysisResponse,
    "analysis"
  > & {
    analysis:
      RoutedUnifiedAnalysisReportView;
  };

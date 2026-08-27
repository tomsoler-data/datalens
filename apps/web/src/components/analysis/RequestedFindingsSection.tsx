"use client";

import type {
  DocumentSummaryView,
  ReportAvailableAnalysisDetailView,
  RequestedPlanView,
  RequestedTimeGranularity,
} from "./analysisTypes";

import type {
  FindingRagContext,
  ReportRequestedFinding,
} from "../../app/types";

import DocumentRequestsSummary
  from "./DocumentRequestsSummary";

import RequestedFindingCard
  from "./RequestedFindingCard";

import { requestedFindingFromAvailableAnalysis }
  from "./requestedAnalysisResolution";

import styles
  from "../../app/page.module.css";


type RequestedFindingsSectionProps = {
  documentSummary: DocumentSummaryView | null;
  requestedPlan: RequestedPlanView | null;
  requestedFindings: ReadonlyArray<ReportRequestedFinding>;
  reportAvailableAnalysisById: Map<string, ReportAvailableAnalysisDetailView>;
  ragContextByAnalysisId: Map<string, FindingRagContext>;
  requestedResolutionErrors: Record<string, string>;
  requestedResolutionLoadingId: string | null;
  handleReconfigureRequestedTimeSeries: (
    analysis: ReportAvailableAnalysisDetailView,
    timeGranularity: RequestedTimeGranularity,
    movingAverageWindow: number
  ) => Promise<void>;
};


export default function RequestedFindingsSection({
  documentSummary,
  requestedPlan,
  requestedFindings,
  reportAvailableAnalysisById,
  ragContextByAnalysisId,
  requestedResolutionErrors,
  requestedResolutionLoadingId,
  handleReconfigureRequestedTimeSeries,
}: RequestedFindingsSectionProps) {
  return (
<details
                                      className={
                                        styles.analysisDisclosure
                                      }
                                    >
                                      <summary
                                        className={
                                          styles.analysisDisclosureSummary
                                        }
                                      >
                                        <div>
                                          <span
                                            className={
                                              styles.eyebrow
                                            }
                                          >
                                            Documentation métier
                                          </span>

                                          <strong>
                                            Demandes issues des documents
                                          </strong>

                                          <small>
                                            Cadrage, demandes vérifiées et résultats
                                            explicitement demandés dans vos documents.
                                          </small>
                                        </div>

                                        <span
                                          className={
                                            styles.analysisDisclosureCount
                                          }
                                        >
                                          {
                                            requestedFindings.length
                                          }
                                        </span>
                                      </summary>

                                      <div
                                        className={
                                          styles.analysisDisclosureBody
                                        }
                                      >
                                        <DocumentRequestsSummary
                                          summary={
                                            documentSummary
                                          }
                                          plan={
                                            requestedPlan
                                          }
                                        />

                                        {
                                          requestedFindings.length >
                                          0
                                            ? (
                                                <div
                                                  className={
                                                    styles.explanationGrid
                                                  }
                                                >
                                                  {
                                                    requestedFindings
                                                      .map(
                                                        (
                                                          finding,
                                                          index
                                                        ) => (
                                                          <RequestedFindingCard
                                                            finding={
                                                              requestedFindingFromAvailableAnalysis(
                                                                reportAvailableAnalysisById.get(
                                                                  finding.analysis_id
                                                                )
                                                              ) ??
                                                              finding
                                                            }
                                                            index={
                                                              index
                                                            }
                                                            ragContext={
                                                              ragContextByAnalysisId.get(
                                                                finding.analysis_id
                                                              ) ??
                                                              null
                                                            }
                                                            key={
                                                              finding.analysis_id
                                                            }

                                                            reconfigurationAnalysis={
                                                              reportAvailableAnalysisById.get(
                                                                finding.analysis_id
                                                              ) ??
                                                              null
                                                            }
                                                            reconfigurationLoading={
                                                              requestedResolutionLoadingId ===
                                                              finding.analysis_id
                                                            }
                                                            reconfigurationError={
                                                              requestedResolutionErrors[
                                                                finding.analysis_id
                                                              ] ??
                                                              null
                                                            }
                                                            onReconfigureTimeSeries={
                                                              handleReconfigureRequestedTimeSeries
                                                            }
/>
                                                        )
                                                      )
                                                  }
                                                </div>
                                              )
                                            : null
                                        }
                                      </div>
                                    </details>
  );
}

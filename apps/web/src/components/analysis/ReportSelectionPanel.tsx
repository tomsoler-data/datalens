"use client";

import type {
  FindingRagContext,
} from "../../app/types";

import styles from "../../app/page.module.css";

import NativeRequestedAnalysisCard
  from "./NativeRequestedAnalysisCard";

import RequestedFindingCard
  from "./RequestedFindingCard";

import type {
  ReportAvailableAnalysisDetailView,
  ReportPromptAnalysisView,
  RequestedTimeGranularity,
} from "./analysisTypes";

import {
  reportSourceLabel,
  requestedLifecycleReasons,
  requestedLifecycleSource,
  requestedLifecycleStatusLabel,
} from "./reportSelectionPresentation";

import {
  requestedLifecycleForAnalysis,
  requestedRankingResolutionAvailability,
  requestedTimeSeriesResolutionAvailability,
} from "./requestedAnalysisResolution";

import type {
  RequestedRankingMetric,
} from "./requestedAnalysisResolution";


type ReportSelectionPanelProps = {
  ragContextByAnalysisId:
    Map<string, FindingRagContext>;

  reportAvailableAnalysisById:
    Map<string, ReportAvailableAnalysisDetailView>;

  reportSelectionLoading:
    boolean;

  requestedResolutionErrors:
    Record<string, string>;

  requestedResolutionLoadingId:
    string | null;

  selectedPromptAnalyses:
    ReportPromptAnalysisView[];

  unresolvedDocumentRequests:
    ReportAvailableAnalysisDetailView[];

  unselectedAutomaticAnalyses:
    ReportAvailableAnalysisDetailView[];

  unselectedRequestedAnalyses:
    ReportAvailableAnalysisDetailView[];

  handleReconfigureRequestedTimeSeries: (
    analysis: ReportAvailableAnalysisDetailView,
    timeGranularity: RequestedTimeGranularity,
    movingAverageWindow: number
  ) => Promise<void>;

  handleResolveRequestedRanking: (
    analysis: ReportAvailableAnalysisDetailView,
    rankingMetric: RequestedRankingMetric
  ) => Promise<void>;

  handleResolveRequestedTimeSeries: (
    analysis: ReportAvailableAnalysisDetailView,
    timeGranularity: RequestedTimeGranularity,
    movingAverageWindow: number
  ) => Promise<void>;

  removePromptAnalysisFromReport: (
    analysisId: string
  ) => Promise<void>;

  setAvailableAnalysisReportSelection: ({
    analysis,
    included,
  }: {
    analysis: ReportAvailableAnalysisDetailView;
    included: boolean;
  }) => Promise<void>;
};


export default function ReportSelectionPanel({
  ragContextByAnalysisId,
  reportAvailableAnalysisById,
  reportSelectionLoading,
  requestedResolutionErrors,
  requestedResolutionLoadingId,
  selectedPromptAnalyses,
  unresolvedDocumentRequests,
  unselectedAutomaticAnalyses,
  unselectedRequestedAnalyses,
  handleReconfigureRequestedTimeSeries,
  handleResolveRequestedRanking,
  handleResolveRequestedTimeSeries,
  removePromptAnalysisFromReport,
  setAvailableAnalysisReportSelection,
}: ReportSelectionPanelProps) {
  return (
<section
                            style={{
                              marginTop:
                                "18px",

                              padding:
                                "20px",

                              border:
                                "1px solid rgba(126, 177, 255, 0.18)",

                              borderRadius:
                                "16px",

                              background:
                                "rgba(3, 8, 17, 0.38)",
                            }}
                            aria-labelledby="prompt-report-selection-title"
                          >
                            <div
                              style={{
                                display:
                                  "flex",

                                alignItems:
                                  "flex-start",

                                justifyContent:
                                  "space-between",

                                gap:
                                  "20px",
                              }}
                            >
                              <div>
                                <span
                                  className={
                                    styles.eyebrow
                                  }
                                >
                                  Sélection du rapport
                                </span>

                                <h2
                                  id="prompt-report-selection-title"
                                  style={{
                                    margin:
                                      "7px 0 0",

                                    color:
                                      "#eef4fc",

                                    fontSize:
                                      "1.15rem",

                                    fontWeight:
                                      600,
                                  }}
                                >
                                  Analyses sélectionnées
                                </h2>

                                <p
                                  className={
                                    styles.resultSubtitle
                                  }
                                  style={{
                                    marginBottom:
                                      0,
                                  }}
                                >
                                  Les analyses demandées par prompt ou dans vos
                                  documents sont sélectionnées par défaut. Vous
                                  pouvez les retirer ; les analyses automatiques
                                  restent facultatives.
                                </p>
                              </div>

                              <span
                                style={{
                                  flex:
                                    "0 0 auto",

                                  padding:
                                    "6px 9px",

                                  border:
                                    "1px solid rgba(122, 203, 160, 0.22)",

                                  borderRadius:
                                    "999px",

                                  color:
                                    "#a4dec2",

                                  background:
                                    "rgba(122, 203, 160, 0.04)",

                                  fontSize:
                                    "0.67rem",

                                  fontWeight:
                                    700,
                                }}
                              >
                                {
                                  selectedPromptAnalyses.length
                                }
                                {" sélectionnée"}
                                {
                                  selectedPromptAnalyses.length >
                                  1
                                    ? "s"
                                    : ""
                                }
                              </span>
                            </div>


                            {
                              selectedPromptAnalyses.length ===
                              0
                                ? (
                                    <div
                                      style={{
                                        marginTop:
                                          "16px",

                                        padding:
                                          "14px 15px",

                                        border:
                                          "1px dashed rgba(154, 174, 204, 0.15)",

                                        borderRadius:
                                          "11px",

                                        color:
                                          "#9eacc0",

                                        fontSize:
                                          "0.76rem",

                                        lineHeight:
                                          1.65,
                                      }}
                                    >
                                      Aucune analyse n’est sélectionnée.
                                      Ajoutez une analyse automatique ou relancez
                                      une demande afin de construire le rapport.
                                    </div>
                                  )
                                : (
                                    <div
                                      style={{
                                        display:
                                          "grid",

                                        gap:
                                          "14px",

                                        marginTop:
                                          "18px",
                                      }}
                                    >
                                      {
                                        selectedPromptAnalyses.map(
                                          (
                                            selectedAnalysis
                                          ) => (
                                            <article
                                              key={
                                                selectedAnalysis.id
                                              }
                                              style={{
                                                display:
                                                  "grid",

                                                gap:
                                                  "9px",
                                              }}
                                            >
                                              <div
                                                style={{
                                                  display:
                                                    "flex",

                                                  alignItems:
                                                    "center",

                                                  justifyContent:
                                                    "space-between",

                                                  gap:
                                                    "12px",

                                                  padding:
                                                    "0 2px",
                                                }}
                                              >
                                                <div>
                                                  <span
                                                    style={{
                                                      display:
                                                        "block",

                                                      color:
                                                        "#92a8c5",

                                                      fontSize:
                                                        "0.63rem",

                                                      fontWeight:
                                                        720,

                                                      letterSpacing:
                                                        "0.055em",

                                                      textTransform:
                                                        "uppercase",
                                                    }}
                                                  >
                                                    {
                                                      selectedAnalysis
                                                        .source_label
                                                    }
                                                  </span>

                                                  <strong
                                                    style={{
                                                      display:
                                                        "block",

                                                      marginTop:
                                                        "4px",

                                                      color:
                                                        "#dce6f3",

                                                      fontSize:
                                                        "0.78rem",

                                                      fontWeight:
                                                        620,
                                                    }}
                                                  >
                                                    {
                                                      selectedAnalysis
                                                        .objective
                                                    }
                                                  </strong>
                                                </div>

                                                <button
                                                  type="button"
                                                  disabled={
                                                    reportSelectionLoading
                                                  }
                                                  onClick={
                                                    () =>
                                                      removePromptAnalysisFromReport(
                                                        selectedAnalysis.id
                                                      )
                                                  }
                                                  style={{
                                                    flex:
                                                      "0 0 auto",

                                                    minHeight:
                                                      "32px",

                                                    padding:
                                                      "0 10px",

                                                    border:
                                                      "1px solid rgba(211, 138, 138, 0.18)",

                                                    borderRadius:
                                                      "9px",

                                                    color:
                                                      "#d7a6a6",

                                                    background:
                                                      "rgba(211, 138, 138, 0.04)",

                                                    font:
                                                      "inherit",

                                                    fontSize:
                                                      "0.67rem",

                                                    fontWeight:
                                                      700,

                                                    cursor:
                                                      reportSelectionLoading
                                                        ? "wait"
                                                        : "pointer",
                                                  }}
                                                >
                                                  Retirer
                                                </button>
                                              </div>

                                              {
                                                selectedAnalysis
                                                  .source_type ===
                                                  "document_request" &&
                                                selectedAnalysis
                                                  .report
                                                  .requested_finding
                                                  ? (
                                                      <RequestedFindingCard
                                                        finding={
                                                          selectedAnalysis
                                                            .report
                                                            .requested_finding
                                                        }
                                                        index={
                                                          Math.max(
                                                            0,
                                                            selectedPromptAnalyses
                                                              .filter(
                                                                (
                                                                  analysis
                                                                ) =>
                                                                  analysis
                                                                    .source_type ===
                                                                  "document_request"
                                                              )
                                                              .findIndex(
                                                                (
                                                                  analysis
                                                                ) =>
                                                                  analysis.id ===
                                                                  selectedAnalysis.id
                                                              )
                                                          )
                                                        }
                                                        ragContext={
                                                          ragContextByAnalysisId.get(
                                                            selectedAnalysis
                                                              .report
                                                              .requested_finding
                                                              .analysis_id
                                                          ) ??
                                                          null
                                                        }

                                                        reconfigurationAnalysis={
                                                          reportAvailableAnalysisById.get(
                                                            selectedAnalysis.id
                                                          ) ??
                                                          null
                                                        }
                                                        reconfigurationLoading={
                                                          requestedResolutionLoadingId ===
                                                          selectedAnalysis.id
                                                        }
                                                        reconfigurationError={
                                                          requestedResolutionErrors[
                                                            selectedAnalysis.id
                                                          ] ??
                                                          null
                                                        }
                                                        onReconfigureTimeSeries={
                                                          handleReconfigureRequestedTimeSeries
                                                        }
/>
                                                    )
                                                  : (
                                                      <NativeRequestedAnalysisCard
                                                        report={
                                                          selectedAnalysis.report
                                                        }
                                                        objective={
                                                          selectedAnalysis.objective
                                                        }
                                                      />
                                                    )
                                              }
                                            </article>
                                          )
                                        )
                                      }
                                    </div>
                                  )
                            }


                            {
                              unresolvedDocumentRequests.length >
                              0
                                ? (
                                    <div
                                      style={{
                                        marginTop:
                                          "18px",

                                        paddingTop:
                                          "16px",

                                        borderTop:
                                          "1px solid rgba(154, 174, 204, 0.10)",
                                      }}
                                    >
                                      <div>
                                        <span
                                          className={
                                            styles.eyebrow
                                          }
                                        >
                                          {
                                            "Demandes n\u00e9cessitant votre attention"
                                          }
                                        </span>

                                        <strong
                                          style={{
                                            display:
                                              "block",

                                            marginTop:
                                              "4px",

                                            color:
                                              "#dce6f3",

                                            fontSize:
                                              "0.82rem",

                                            lineHeight:
                                              1.5,
                                          }}
                                        >
                                          {
                                            `${unresolvedDocumentRequests.length} demande(s) n'ont pas \u00e9t\u00e9 ex\u00e9cut\u00e9es`
                                          }
                                        </strong>

                                        <p
                                          style={{
                                            margin:
                                              "6px 0 0",

                                            maxWidth:
                                              "760px",

                                            color:
                                              "#92a8c5",

                                            fontSize:
                                              "0.72rem",

                                            lineHeight:
                                              1.55,
                                          }}
                                        >
                                          {
                                            "DataLens conserve ces demandes pour la tra\u00e7abilit\u00e9, mais ne les pr\u00e9sente pas comme des r\u00e9sultats analytiques."
                                          }
                                        </p>
                                      </div>


                                      <div
                                        style={{
                                          display:
                                            "grid",

                                          gap:
                                            "10px",

                                          marginTop:
                                            "12px",
                                        }}
                                      >
                                        {
                                          unresolvedDocumentRequests.map(
                                            (
                                              analysis
                                            ) => {
                                              const lifecycle =
                                                requestedLifecycleForAnalysis(
                                                  analysis
                                                );


                                              if (
                                                lifecycle ===
                                                null
                                              ) {
                                                return null;
                                              }


                                              const planStatus =
                                                (
                                                  lifecycle
                                                    .plan_status ??
                                                  ""
                                                )
                                                  .trim()
                                                  .toLowerCase();


                                              const isBlocked =
                                                planStatus ===
                                                "blocked";


                                              const reasons =
                                                requestedLifecycleReasons(
                                                  lifecycle
                                                );


                                              const source =
                                                requestedLifecycleSource(
                                                  lifecycle
                                                );


                                              const requestText =
                                                (
                                                  lifecycle
                                                    .request_text ??
                                                  analysis
                                                    .objective
                                                )
                                                  .trim();


                                              const rankingResolution =
                                                requestedRankingResolutionAvailability(
                                                  analysis
                                                );


                                              const requestId =
                                                rankingResolution
                                                  .requestId;


                                              const isResolvableRanking =
                                                planStatus ===
                                                  "ambiguous" &&
                                                rankingResolution
                                                  .isRankingRequest &&
                                                Boolean(
                                                  requestId
                                                );


                                              const timeSeriesResolution =
                                                requestedTimeSeriesResolutionAvailability(
                                                  analysis
                                                );


                                              const timeSeriesRequestId =
                                                timeSeriesResolution
                                                  .requestId;


                                              const isResolvableTimeSeries =
                                                planStatus ===
                                                  "ambiguous" &&
                                                timeSeriesResolution
                                                  .isTimeSeriesRequest &&
                                                timeSeriesResolution
                                                  .executableInputs &&
                                                Boolean(
                                                  timeSeriesRequestId
                                                );


                                              const resolutionLoading =
                                                requestedResolutionLoadingId ===
                                                analysis.analysis_id;


                                              const resolutionError =
                                                requestedResolutionErrors[
                                                  analysis.analysis_id
                                                ] ??
                                                null;



                                              return (
                                                <article
                                                  key={
                                                    analysis.analysis_id
                                                  }
                                                  style={{
                                                    padding:
                                                      "13px 14px",

                                                    border:
                                                      isBlocked
                                                        ? "1px solid rgba(226, 112, 112, 0.22)"
                                                        : "1px solid rgba(222, 177, 93, 0.22)",

                                                    borderRadius:
                                                      "10px",

                                                    background:
                                                      isBlocked
                                                        ? "rgba(137, 48, 48, 0.06)"
                                                        : "rgba(151, 109, 36, 0.06)",
                                                  }}
                                                >
                                                  <div
                                                    style={{
                                                      display:
                                                        "flex",

                                                      alignItems:
                                                        "center",

                                                      flexWrap:
                                                        "wrap",

                                                      gap:
                                                        "8px",
                                                    }}
                                                  >
                                                    <span
                                                      style={{
                                                        display:
                                                          "inline-flex",

                                                        alignItems:
                                                          "center",

                                                        minHeight:
                                                          "24px",

                                                        padding:
                                                          "0 8px",

                                                        borderRadius:
                                                          "999px",

                                                        border:
                                                          isBlocked
                                                            ? "1px solid rgba(226, 112, 112, 0.28)"
                                                            : "1px solid rgba(222, 177, 93, 0.28)",

                                                        color:
                                                          isBlocked
                                                            ? "#efaaaa"
                                                            : "#e6c47e",

                                                        background:
                                                          isBlocked
                                                            ? "rgba(226, 112, 112, 0.07)"
                                                            : "rgba(222, 177, 93, 0.07)",

                                                        fontSize:
                                                          "0.61rem",

                                                        fontWeight:
                                                          760,

                                                        letterSpacing:
                                                          "0.055em",

                                                        textTransform:
                                                          "uppercase",
                                                      }}
                                                    >
                                                      {
                                                        requestedLifecycleStatusLabel(
                                                          lifecycle
                                                            .plan_status
                                                        )
                                                      }
                                                    </span>

                                                    <span
                                                      style={{
                                                        color:
                                                          "#7890ad",

                                                        fontSize:
                                                          "0.63rem",

                                                        fontWeight:
                                                          650,
                                                      }}
                                                    >
                                                      {
                                                        "Demande du document"
                                                      }
                                                    </span>
                                                  </div>


                                                  <strong
                                                    style={{
                                                      display:
                                                        "block",

                                                      marginTop:
                                                        "9px",

                                                      color:
                                                        "#e4edf8",

                                                      fontSize:
                                                        "0.78rem",

                                                      fontWeight:
                                                        650,

                                                      lineHeight:
                                                        1.5,
                                                    }}
                                                  >
                                                    {
                                                      requestText
                                                    }
                                                  </strong>


                                                  {
                                                    reasons.map(
                                                      (
                                                        reason,
                                                        index
                                                      ) => (
                                                        <p
                                                          key={
                                                            `${analysis.analysis_id}-reason-${index}`
                                                          }
                                                          style={{
                                                            margin:
                                                              index ===
                                                              0
                                                                ? "7px 0 0"
                                                                : "4px 0 0",

                                                            color:
                                                              "#aebdd0",

                                                            fontSize:
                                                              "0.7rem",

                                                            lineHeight:
                                                              1.55,
                                                          }}
                                                        >
                                                          {
                                                            index ===
                                                            0
                                                              ? `Motif : ${reason}`
                                                              : reason
                                                          }
                                                        </p>
                                                      )
                                                    )
                                                  }


                                                  {
                                                    source
                                                      ? (
                                                          <span
                                                            style={{
                                                              display:
                                                                "block",

                                                              marginTop:
                                                                "8px",

                                                              color:
                                                                "#7187a3",

                                                              fontSize:
                                                                "0.63rem",

                                                              lineHeight:
                                                                1.45,
                                                            }}
                                                          >
                                                            {
                                                              `Source : ${source}`
                                                            }
                                                          </span>
                                                        )
                                                      : null
                                                  }



                                                  {
                                                    isResolvableTimeSeries
                                                      ? (
                                                          <form
                                                            onSubmit={
                                                              (
                                                                event
                                                              ) => {
                                                                event.preventDefault();


                                                                const formData =
                                                                  new FormData(
                                                                    event.currentTarget
                                                                  );


                                                                const rawGranularity =
                                                                  String(
                                                                    formData.get(
                                                                      "time_granularity"
                                                                    ) ??
                                                                    ""
                                                                  );


                                                                const allowed:
                                                                  RequestedTimeGranularity[] =
                                                                    [
                                                                      "day",
                                                                      "week",
                                                                      "month",
                                                                      "quarter",
                                                                      "year",
                                                                    ];


                                                                if (
                                                                  !allowed.includes(
                                                                    rawGranularity as RequestedTimeGranularity
                                                                  )
                                                                ) {
                                                                  return;
                                                                }


                                                                const windowValue =
                                                                  Number(
                                                                    formData.get(
                                                                      "moving_average_window"
                                                                    )
                                                                  );


                                                                if (
                                                                  !Number.isInteger(
                                                                    windowValue
                                                                  ) ||
                                                                  windowValue <
                                                                    1
                                                                ) {
                                                                  return;
                                                                }


                                                                void handleResolveRequestedTimeSeries(
                                                                  analysis,
                                                                  rawGranularity as RequestedTimeGranularity,
                                                                  windowValue
                                                                );
                                                              }
                                                            }
                                                            style={{
                                                              marginTop:
                                                                "12px",

                                                              paddingTop:
                                                                "12px",

                                                              borderTop:
                                                                "1px solid rgba(222, 177, 93, 0.14)",
                                                            }}
                                                          >
                                                            <strong
                                                              style={{
                                                                display:
                                                                  "block",

                                                                color:
                                                                  "#dce6f3",

                                                                fontSize:
                                                                  "0.72rem",

                                                                fontWeight:
                                                                  650,

                                                                lineHeight:
                                                                  1.45,
                                                              }}
                                                            >
                                                              {
                                                                "Choisissez la granularit\u00e9 temporelle"
                                                              }
                                                            </strong>


                                                            <p
                                                              style={{
                                                                margin:
                                                                  "4px 0 0",

                                                                color:
                                                                  "#92a8c5",

                                                                fontSize:
                                                                  "0.68rem",

                                                                lineHeight:
                                                                  1.5,
                                                              }}
                                                            >
                                                              {
                                                                "DataLens recalculera le chiffre d'affaires et la moyenne mobile depuis les donn\u00e9es valid\u00e9es. Le navigateur ne choisit ni les colonnes ni le dataset."
                                                              }
                                                            </p>


                                                            <div
                                                              style={{
                                                                display:
                                                                  "grid",

                                                                gridTemplateColumns:
                                                                  "minmax(160px, 220px) minmax(130px, 180px)",

                                                                gap:
                                                                  "10px",

                                                                marginTop:
                                                                  "11px",

                                                                alignItems:
                                                                  "end",
                                                              }}
                                                            >
                                                              <label
                                                                style={{
                                                                  display:
                                                                    "grid",

                                                                  gap:
                                                                    "5px",

                                                                  color:
                                                                    "#aebdd0",

                                                                  fontSize:
                                                                    "0.66rem",
                                                                }}
                                                              >
                                                                {
                                                                  "P\u00e9riode"
                                                                }

                                                                <select
                                                                  name="time_granularity"
                                                                  defaultValue="month"
                                                                  disabled={
                                                                    resolutionLoading
                                                                  }
                                                                  style={{
                                                                    minHeight:
                                                                      "36px",

                                                                    padding:
                                                                      "0 9px",

                                                                    border:
                                                                      "1px solid rgba(116, 177, 255, 0.24)",

                                                                    borderRadius:
                                                                      "8px",

                                                                    background:
                                                                      "rgba(8, 23, 34, 0.85)",

                                                                    color:
                                                                      "#dce6f3",
                                                                  }}
                                                                >
                                                                  <option value="day">
                                                                    Jour
                                                                  </option>

                                                                  <option value="week">
                                                                    Semaine
                                                                  </option>

                                                                  <option value="month">
                                                                    Mois
                                                                  </option>

                                                                  <option value="quarter">
                                                                    Trimestre
                                                                  </option>

                                                                  <option value="year">
                                                                    {
                                                                      "Ann\u00e9e"
                                                                    }
                                                                  </option>
                                                                </select>
                                                              </label>


                                                              <label
                                                                style={{
                                                                  display:
                                                                    "grid",

                                                                  gap:
                                                                    "5px",

                                                                  color:
                                                                    "#aebdd0",

                                                                  fontSize:
                                                                    "0.66rem",
                                                                }}
                                                              >
                                                                {
                                                                  "Fen\u00eatre mobile"
                                                                }

                                                                <input
                                                                  name="moving_average_window"
                                                                  type="number"
                                                                  min={
                                                                    1
                                                                  }
                                                                  step={
                                                                    1
                                                                  }
                                                                  defaultValue={
                                                                    3
                                                                  }
                                                                  required
                                                                  disabled={
                                                                    resolutionLoading
                                                                  }
                                                                  style={{
                                                                    minHeight:
                                                                      "36px",

                                                                    padding:
                                                                      "0 9px",

                                                                    border:
                                                                      "1px solid rgba(116, 177, 255, 0.24)",

                                                                    borderRadius:
                                                                      "8px",

                                                                    background:
                                                                      "rgba(8, 23, 34, 0.85)",

                                                                    color:
                                                                      "#dce6f3",
                                                                  }}
                                                                />
                                                              </label>
                                                            </div>


                                                            <button
                                                              type="submit"
                                                              disabled={
                                                                resolutionLoading
                                                              }
                                                              style={{
                                                                minHeight:
                                                                  "36px",

                                                                marginTop:
                                                                  "10px",

                                                                padding:
                                                                  "0 12px",

                                                                border:
                                                                  "1px solid rgba(116, 177, 255, 0.28)",

                                                                borderRadius:
                                                                  "8px",

                                                                background:
                                                                  "rgba(59, 119, 196, 0.10)",

                                                                color:
                                                                  "#d9e9ff",

                                                                cursor:
                                                                  resolutionLoading
                                                                    ? "wait"
                                                                    : "pointer",

                                                                opacity:
                                                                  resolutionLoading
                                                                    ? 0.65
                                                                    : 1,

                                                                fontSize:
                                                                  "0.68rem",

                                                                fontWeight:
                                                                  650,
                                                              }}
                                                            >
                                                              {
                                                                resolutionLoading
                                                                  ? (
                                                                      "Calcul en cours..."
                                                                    )
                                                                  : (
                                                                      "Calculer avec ces param\u00e8tres"
                                                                    )
                                                              }
                                                            </button>


                                                            {
                                                              resolutionError
                                                                ? (
                                                                    <div
                                                                      role="alert"
                                                                      style={{
                                                                        marginTop:
                                                                          "8px",

                                                                        padding:
                                                                          "8px 9px",

                                                                        border:
                                                                          "1px solid rgba(226, 112, 112, 0.20)",

                                                                        borderRadius:
                                                                          "7px",

                                                                        background:
                                                                          "rgba(137, 48, 48, 0.06)",

                                                                        color:
                                                                          "#efaaaa",

                                                                        fontSize:
                                                                          "0.66rem",

                                                                        lineHeight:
                                                                          1.5,
                                                                      }}
                                                                    >
                                                                      {
                                                                        resolutionError
                                                                      }
                                                                    </div>
                                                                  )
                                                                : null
                                                            }
                                                          </form>
                                                        )
                                                      : null
                                                  }


                                                  {
                                                    isResolvableRanking
                                                      ? (
                                                          <div
                                                            style={{
                                                              marginTop:
                                                                "12px",

                                                              paddingTop:
                                                                "12px",

                                                              borderTop:
                                                                "1px solid rgba(222, 177, 93, 0.14)",
                                                            }}
                                                          >
                                                            <strong
                                                              style={{
                                                                display:
                                                                  "block",

                                                                color:
                                                                  "#dce6f3",

                                                                fontSize:
                                                                  "0.72rem",

                                                                fontWeight:
                                                                  650,

                                                                lineHeight:
                                                                  1.45,
                                                              }}
                                                            >
                                                              Choisissez la métrique de classement
                                                            </strong>


                                                            <p
                                                              style={{
                                                                margin:
                                                                  "4px 0 0",

                                                                color:
                                                                  "#92a8c5",

                                                                fontSize:
                                                                  "0.68rem",

                                                                lineHeight:
                                                                  1.5,
                                                              }}
                                                            >
                                                              {
                                                                "Votre choix est envoyé comme clarification uniquement. Les colonnes, le dataset et le calcul restent contrôlés par DataLens."
                                                              }
                                                            </p>


                                                            <div
                                                              style={{
                                                                display:
                                                                  "flex",

                                                                flexWrap:
                                                                  "wrap",

                                                                gap:
                                                                  "8px",

                                                                marginTop:
                                                                  "10px",
                                                              }}
                                                            >
                                                              <button
                                                                type="button"
                                                                disabled={
                                                                  resolutionLoading ||
                                                                  !rankingResolution
                                                                    .revenue
                                                                }
                                                                onClick={
                                                                  () => {
                                                                    void handleResolveRequestedRanking(
                                                                      analysis,
                                                                      "revenue"
                                                                    );
                                                                  }
                                                                }
                                                                style={{
                                                                  minHeight:
                                                                    "34px",

                                                                  padding:
                                                                    "0 11px",

                                                                  border:
                                                                    "1px solid rgba(116, 177, 255, 0.28)",

                                                                  borderRadius:
                                                                    "8px",

                                                                  background:
                                                                    "rgba(59, 119, 196, 0.10)",

                                                                  color:
                                                                    rankingResolution
                                                                      .revenue
                                                                      ? "#d9e9ff"
                                                                      : "#718197",

                                                                  cursor:
                                                                    resolutionLoading ||
                                                                    !rankingResolution
                                                                      .revenue
                                                                      ? "not-allowed"
                                                                      : "pointer",

                                                                  opacity:
                                                                    resolutionLoading ||
                                                                    !rankingResolution
                                                                      .revenue
                                                                      ? 0.55
                                                                      : 1,

                                                                  fontSize:
                                                                    "0.68rem",

                                                                  fontWeight:
                                                                    650,
                                                                }}
                                                              >
                                                                Chiffre d'affaires
                                                              </button>


                                                              <button
                                                                type="button"
                                                                disabled={
                                                                  resolutionLoading ||
                                                                  !rankingResolution
                                                                    .transactionCount
                                                                }
                                                                onClick={
                                                                  () => {
                                                                    void handleResolveRequestedRanking(
                                                                      analysis,
                                                                      "transaction_count"
                                                                    );
                                                                  }
                                                                }
                                                                style={{
                                                                  minHeight:
                                                                    "34px",

                                                                  padding:
                                                                    "0 11px",

                                                                  border:
                                                                    "1px solid rgba(116, 177, 255, 0.28)",

                                                                  borderRadius:
                                                                    "8px",

                                                                  background:
                                                                    "rgba(59, 119, 196, 0.10)",

                                                                  color:
                                                                    rankingResolution
                                                                      .transactionCount
                                                                      ? "#d9e9ff"
                                                                      : "#718197",

                                                                  cursor:
                                                                    resolutionLoading ||
                                                                    !rankingResolution
                                                                      .transactionCount
                                                                      ? "not-allowed"
                                                                      : "pointer",

                                                                  opacity:
                                                                    resolutionLoading ||
                                                                    !rankingResolution
                                                                      .transactionCount
                                                                      ? 0.55
                                                                      : 1,

                                                                  fontSize:
                                                                    "0.68rem",

                                                                  fontWeight:
                                                                    650,
                                                                }}
                                                              >
                                                                Nombre de transactions
                                                              </button>
                                                            </div>


                                                            <div
                                                              style={{
                                                                marginTop:
                                                                  "8px",

                                                                color:
                                                                  "#72859e",

                                                                fontSize:
                                                                  "0.64rem",

                                                                lineHeight:
                                                                  1.45,
                                                              }}
                                                            >
                                                              {
                                                                "Volume vendu : indisponible tant qu'aucune quantité fiable n'est résolue."
                                                              }
                                                            </div>


                                                            {
                                                              resolutionLoading
                                                                ? (
                                                                    <div
                                                                      aria-live="polite"
                                                                      style={{
                                                                        marginTop:
                                                                          "8px",

                                                                        color:
                                                                          "#9eb7d6",

                                                                        fontSize:
                                                                          "0.66rem",
                                                                      }}
                                                                    >
                                                                      Résolution et calcul déterministe en cours…
                                                                    </div>
                                                                  )
                                                                : null
                                                            }


                                                            {
                                                              resolutionError
                                                                ? (
                                                                    <div
                                                                      role="alert"
                                                                      style={{
                                                                        marginTop:
                                                                          "8px",

                                                                        padding:
                                                                          "8px 9px",

                                                                        border:
                                                                          "1px solid rgba(226, 112, 112, 0.20)",

                                                                        borderRadius:
                                                                          "7px",

                                                                        background:
                                                                          "rgba(137, 48, 48, 0.06)",

                                                                        color:
                                                                          "#efaaaa",

                                                                        fontSize:
                                                                          "0.66rem",

                                                                        lineHeight:
                                                                          1.5,
                                                                      }}
                                                                    >
                                                                      {
                                                                        resolutionError
                                                                      }
                                                                    </div>
                                                                  )
                                                                : null
                                                            }
                                                          </div>
                                                        )
                                                      : null
                                                  }
                                                </article>
                                              );
                                            }
                                          )
                                        }
                                      </div>
                                    </div>
                                  )
                                : null
                            }


                            {
                              unselectedRequestedAnalyses.length >
                              0
                                ? (
                                    <div
                                      style={{
                                        marginTop:
                                          "18px",

                                        paddingTop:
                                          "16px",

                                        borderTop:
                                          "1px solid rgba(154, 174, 204, 0.10)",
                                      }}
                                    >
                                      <div>
                                        <span
                                          className={
                                            styles.eyebrow
                                          }
                                        >
                                          Demandes retirées
                                        </span>

                                        <strong
                                          style={{
                                            display:
                                              "block",

                                            marginTop:
                                              "4px",

                                            fontSize:
                                              "0.82rem",
                                          }}
                                        >
                                          Réintégrer une analyse demandée
                                        </strong>
                                      </div>

                                      <div
                                        style={{
                                          display:
                                            "grid",

                                          gap:
                                            "8px",

                                          marginTop:
                                            "12px",
                                        }}
                                      >
                                        {
                                          unselectedRequestedAnalyses.map(
                                            (
                                              analysis
                                            ) => (
                                              <article
                                                key={
                                                  analysis.analysis_id
                                                }
                                                style={{
                                                  display:
                                                    "grid",

                                                  gridTemplateColumns:
                                                    "minmax(0, 1fr) auto",

                                                  alignItems:
                                                    "center",

                                                  gap:
                                                    "14px",

                                                  padding:
                                                    "11px 12px",

                                                  border:
                                                    "1px solid rgba(154, 174, 204, 0.10)",

                                                  borderRadius:
                                                    "10px",

                                                  background:
                                                    "rgba(3, 8, 17, 0.26)",
                                                }}
                                              >
                                                <div
                                                  style={{
                                                    minWidth:
                                                      0,
                                                  }}
                                                >
                                                  <span
                                                    style={{
                                                      display:
                                                        "block",

                                                      color:
                                                        "#92a8c5",

                                                      fontSize:
                                                        "0.61rem",

                                                      fontWeight:
                                                        720,

                                                      letterSpacing:
                                                        "0.055em",

                                                      textTransform:
                                                        "uppercase",
                                                    }}
                                                  >
                                                    {
                                                      reportSourceLabel(
                                                        analysis.source_type
                                                      )
                                                    }
                                                  </span>

                                                  <strong
                                                    style={{
                                                      display:
                                                        "block",

                                                      marginTop:
                                                        "4px",

                                                      overflow:
                                                        "hidden",

                                                      color:
                                                        "#dce6f3",

                                                      fontSize:
                                                        "0.75rem",

                                                      fontWeight:
                                                        620,

                                                      lineHeight:
                                                        1.45,

                                                      textOverflow:
                                                        "ellipsis",

                                                      whiteSpace:
                                                        "nowrap",
                                                    }}
                                                    title={
                                                      analysis.objective
                                                    }
                                                  >
                                                    {
                                                      analysis.objective
                                                    }
                                                  </strong>
                                                </div>

                                                <button
                                                  type="button"
                                                  disabled={
                                                    reportSelectionLoading
                                                  }
                                                  onClick={
                                                    () =>
                                                      void setAvailableAnalysisReportSelection(
                                                        {
                                                          analysis,

                                                          included:
                                                            true,
                                                        }
                                                      )
                                                  }
                                                  style={{
                                                    minHeight:
                                                      "32px",

                                                    padding:
                                                      "0 10px",

                                                    border:
                                                      "1px solid rgba(122, 203, 160, 0.20)",

                                                    borderRadius:
                                                      "9px",

                                                    color:
                                                      "#a4dec2",

                                                    background:
                                                      "rgba(122, 203, 160, 0.04)",

                                                    font:
                                                      "inherit",

                                                    fontSize:
                                                      "0.67rem",

                                                    fontWeight:
                                                      700,

                                                    cursor:
                                                      reportSelectionLoading
                                                        ? "wait"
                                                        : "pointer",
                                                  }}
                                                >
                                                  Réintégrer
                                                </button>
                                              </article>
                                            )
                                          )
                                        }
                                      </div>
                                    </div>
                                  )
                                : null
                            }


                            {
                              unselectedAutomaticAnalyses.length >
                              0
                                ? (
                                    <div
                                      style={{
                                        marginTop:
                                          "18px",

                                        paddingTop:
                                          "16px",

                                        borderTop:
                                          "1px solid rgba(154, 174, 204, 0.10)",
                                      }}
                                    >
                                      <div
                                        style={{
                                          display:
                                            "flex",

                                          alignItems:
                                            "center",

                                          justifyContent:
                                            "space-between",

                                          gap:
                                            "12px",
                                        }}
                                      >
                                        <div>
                                          <span
                                            className={
                                              styles.eyebrow
                                            }
                                          >
                                            Analyses automatiques
                                          </span>

                                          <strong
                                            style={{
                                              display:
                                                "block",

                                              marginTop:
                                                "4px",

                                              fontSize:
                                                "0.82rem",
                                            }}
                                          >
                                            Ajouter seulement ce qui est utile
                                          </strong>
                                        </div>

                                        <span
                                          style={{
                                            color:
                                              "#91a0b5",

                                            fontSize:
                                              "0.67rem",
                                          }}
                                        >
                                          {
                                            unselectedAutomaticAnalyses.length
                                          }
                                          {" disponible"}
                                          {
                                            unselectedAutomaticAnalyses.length >
                                            1
                                              ? "s"
                                              : ""
                                          }
                                        </span>
                                      </div>

                                      <div
                                        style={{
                                          display:
                                            "grid",

                                          gap:
                                            "8px",

                                          marginTop:
                                            "12px",
                                        }}
                                      >
                                        {
                                          unselectedAutomaticAnalyses.map(
                                            (
                                              analysis
                                            ) => (
                                              <article
                                                key={
                                                  analysis.analysis_id
                                                }
                                                style={{
                                                  display:
                                                    "grid",

                                                  gridTemplateColumns:
                                                    "minmax(0, 1fr) auto",

                                                  alignItems:
                                                    "center",

                                                  gap:
                                                    "14px",

                                                  padding:
                                                    "11px 12px",

                                                  border:
                                                    "1px solid rgba(154, 174, 204, 0.10)",

                                                  borderRadius:
                                                    "10px",

                                                  background:
                                                    "rgba(3, 8, 17, 0.26)",
                                                }}
                                              >
                                                <div
                                                  style={{
                                                    minWidth:
                                                      0,
                                                  }}
                                                >
                                                  <span
                                                    style={{
                                                      display:
                                                        "block",

                                                      color:
                                                        "#92a8c5",

                                                      fontSize:
                                                        "0.61rem",

                                                      fontWeight:
                                                        720,

                                                      letterSpacing:
                                                        "0.055em",

                                                      textTransform:
                                                        "uppercase",
                                                    }}
                                                  >
                                                    Analyse automatique
                                                  </span>

                                                  <strong
                                                    style={{
                                                      display:
                                                        "block",

                                                      marginTop:
                                                        "4px",

                                                      overflow:
                                                        "hidden",

                                                      color:
                                                        "#dce6f3",

                                                      fontSize:
                                                        "0.75rem",

                                                      fontWeight:
                                                        620,

                                                      lineHeight:
                                                        1.45,

                                                      textOverflow:
                                                        "ellipsis",

                                                      whiteSpace:
                                                        "nowrap",
                                                    }}
                                                    title={
                                                      analysis.objective
                                                    }
                                                  >
                                                    {
                                                      analysis.objective
                                                    }
                                                  </strong>
                                                </div>

                                                <button
                                                  type="button"
                                                  disabled={
                                                    reportSelectionLoading
                                                  }
                                                  onClick={
                                                    () =>
                                                      void setAvailableAnalysisReportSelection(
                                                        {
                                                          analysis,

                                                          included:
                                                            true,
                                                        }
                                                      )
                                                  }
                                                  style={{
                                                    minHeight:
                                                      "32px",

                                                    padding:
                                                      "0 10px",

                                                    border:
                                                      "1px solid rgba(122, 203, 160, 0.20)",

                                                    borderRadius:
                                                      "9px",

                                                    color:
                                                      "#a4dec2",

                                                    background:
                                                      "rgba(122, 203, 160, 0.04)",

                                                    font:
                                                      "inherit",

                                                    fontSize:
                                                      "0.67rem",

                                                    fontWeight:
                                                      700,

                                                    cursor:
                                                      reportSelectionLoading
                                                        ? "wait"
                                                        : "pointer",
                                                  }}
                                                >
                                                  Ajouter au rapport
                                                </button>
                                              </article>
                                            )
                                          )
                                        }
                                      </div>
                                    </div>
                                  )
                                : null
                            }

                          </section>
  );
}

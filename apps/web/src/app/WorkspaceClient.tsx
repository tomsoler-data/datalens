"use client";

import type {
  RoutedContextualizedAnalysisResponseView,
  RoutedUnifiedAnalysisReportView,
} from "./workspaceAnalysisTypes";


import {
  ChangeEvent,
  FormEvent,
  useEffect,
  useMemo,
  useState,
} from "react";

import type {
  ReactNode,
} from "react";



import PreparationWorkflowPanel from "../components/preparation/PreparationWorkflowPanel";
import PreparationWorkflowHistoryPanel from "../components/preparation/PreparationWorkflowHistoryPanel";
import PreparationFinalizationPanel from "../components/preparation/PreparationFinalizationPanel";
import PreparationUnderstandingPanel from "../components/preparation/PreparationUnderstandingPanel";
import PreparationResolvedStagePanel from "../components/preparation/PreparationResolvedStagePanel";
import CleaningPlanPanel from "../components/preparation/CleaningPlanPanel";
import SemanticReviewPanel from "../components/preparation/SemanticReviewPanel";
import QualityReportSection from "../components/preparation/QualityReportSection";
import DataPreparationStudio from "../components/preparation/DataPreparationStudio";
import PreparationCombinePanel from "../components/preparation/PreparationCombinePanel";
import PreparationSubstepNavigation from "../components/preparation/PreparationSubstepNavigation";
import PreparationTransformPanel from "../components/preparation/PreparationTransformPanel";
import SemanticConfirmationPanel from "../components/preparation/SemanticConfirmationPanel";





import type {
  PreparationSubstep,
} from "../components/preparation/PreparationSubstepNavigation";

import {
  createPreparationSession,
  getPreparationSession,
  PreparationApiError,
  validatePreparationSession,
  getPreparationUiState,
  getPreparationIngestionView,
} from "../components/preparation/preparationApi";
import {
  ingestDatasetFiles,
} from "../components/preparation/datasetIngestionApi";
import {
  runPreparationQuality,
} from "../components/preparation/preparationQualityApi";
import {
  createPreparationPipelineReset,
} from "../components/preparation/preparationResetActions";

import {
  confirmSemanticReview,
  SemanticConfirmationApiError,
} from "../components/preparation/semanticConfirmationApi";
import {
  requestSemanticReview,
} from "../components/preparation/semanticReviewApi";
import {
  applySemanticCleaning,
} from "../components/preparation/semanticCleaningApi";
import {
  applyDeterministicCleaning,
} from "../components/preparation/deterministicCleaningApi";
import {
  prepareDeterministicCleaningApply,
} from "../components/preparation/deterministicCleaningPreparation";



import type {
  CleaningExecutionView,
  CleaningPlanView,
  DataQualityReportView,
  PreparationSessionView,
  PreparationStageRecord,
  SemanticCleaningExecutionView,
  SemanticCleaningPlanView,
} from "../components/preparation/preparationTypes";
import {
  findPreparationStage,
  preparationSubstepFromSession,
  requiresCombineDiscoveryBeforeValidation,
  preparationStageResolved,
} from "../components/preparation/preparationWorkflowHelpers";
import {
  clearActivePreparationWorkflowId,
  persistActivePreparationWorkflowId,
  readActivePreparationWorkflowId,
} from "../components/preparation/preparationWorkflowStorage";
import { formatNumber } from "../components/analysis/analysisPresentation";
import { plannerUiCopy } from "../components/analysis/analysisPlanningPresentation";
import { nativePipelineHasExecutedResult } from "../components/analysis/analysisExecutionPresentation";
import type {
  AIPlannerReportView,
  AINativePipelineReportView,
  ReportSelectionItemView,
  ReportAvailableAnalysisDetailView,
  ReportSelectionDetailsView,
  ReportPromptAnalysisView,
  AnalysisFollowUpTurn,
  DocumentSummaryView,
  RequestedPlanView,
} from "../components/analysis/analysisTypes";
import {
  loadReportSelectionState,
} from "../components/analysis/reportSelectionApi";
import {
  runAiNativeAnalysis,
} from "../components/analysis/aiNativeApi";
import {
  runAnalysisRequest,
} from "../components/analysis/analysisRunApi";
import AnalysisExecutionPanel
  from "../components/analysis/AnalysisExecutionPanel";
import ReportSelectionPanel
  from "../components/analysis/ReportSelectionPanel";
import {
  createAnalysisOutputReset,
} from "../components/analysis/analysisResetActions";

import ExpandableChart from "../components/analysis/charts/ExpandableChart";
import {
  lineBandRenderablePoints,
} from "../components/analysis/charts/LineBandChart";
import FindingChart from "../components/analysis/charts/FindingChart";
import NativeRequestedAnalysisCard from "../components/analysis/NativeRequestedAnalysisCard";
import SelectableNativeAnalysisResult
  from "../components/analysis/SelectableNativeAnalysisResult";
import AnalysisFollowUpHistory
  from "../components/analysis/AnalysisFollowUpHistory";



import MainFindingsSection
  from "../components/analysis/MainFindingsSection";
import RequestedFindingsSection
  from "../components/analysis/RequestedFindingsSection";
import PlannerBlockedAnalysisCard
  from "../components/analysis/PlannerBlockedAnalysisCard";



import { buildSignalKpis }
  from "../components/analysis/analysisSignalKpis";
import WorkspaceNavigation
  from "../components/workspace/WorkspaceNavigation";

import type { WorkspaceStep }
  from "../components/workspace/workspaceNavigationTypes";
import { persistActiveWorkspaceStep } from "../components/workspace/workspaceNavigationStorage";
import RequestedFindingCard from "../components/analysis/RequestedFindingCard";
import EntityOutlierRequestedAnswer from "../components/analysis/EntityOutlierRequestedAnswer";

import styles from "./page.module.css";

import type {
  DatasetManifest,
  FindingRagContext,
  MultiDatasetIngestion,
  RagContextReport,
  ReportChartDatum,
  ReportFinding,
  ReportRequestedFinding,
} from "./types";

import {
  createRequestedAnalysisResolutionHandlers,
  requestedLifecycleForAnalysis,
  requestedFindingFromAvailableAnalysis,
  requestedLifecycleOrder,
} from "../components/analysis/requestedAnalysisResolution";

import type {
  RequestedAnalysisLifecycleView,
} from "../components/analysis/requestedAnalysisResolution";

import { createPreparedDataExportHandler } from "../components/preparation/preparationExport";

import { createReportPdfExportHandler } from "../components/analysis/reportPdfExport";

import { createAvailableAnalysisReportSelection } from "../components/analysis/availableAnalysisReportSelection";

import { createPromptAnalysisReportSelection } from "../components/analysis/promptAnalysisReportSelection";

import { createCleaningPlanLoader } from "../components/preparation/cleaningPlanLoader";

import { useSemanticPreparationState } from "../components/preparation/useSemanticPreparationState";
import { reportSourceLabel } from "../components/analysis/reportSelectionPresentation";

import AnalysisAuditDisclosure from "./AnalysisAuditDisclosure";
import SelectedPromptAnalysesSection from "./SelectedPromptAnalysesSection";
import DatasetWorkspaceSection from "./DatasetWorkspaceSection";
import BusinessDocumentsSection from "./BusinessDocumentsSection";
import AvailableAnalysisSelectionHeader from "./AvailableAnalysisSelectionHeader";
import DatasetImportControls from "./DatasetImportControls";
import ReportSynthesisActions from "./ReportSynthesisActions";
import {
  deriveRestoredAppliedCleaningActionIds,
  deriveRestoredSelectedCleaningActionIds,
  resolveRestoredWorkspaceStep,
} from "./preparationWorkflowRestoration";
import { deriveReportSelectionRestoration } from "./reportSelectionRestoration";


const API_URL =
  process.env.NEXT_PUBLIC_DATALENS_API_URL ??
  "http://127.0.0.1:8000";


















/* ============================================================
   PREPARATION UI STATE REHYDRATION
   PREPARATION_UI_STATE_FRONTEND_V0_1
============================================================ */

























































































/* ============================================================
   ENTITY OUTLIERS — ROUTED USER-FACING FINDING
============================================================ */
























































/* ============================================================
   ANALYSIS_F5_REHYDRATION_V0_2

   Only presentation/navigation state is remembered here.
   Analytical results remain server-owned.
============================================================ */























export default function WorkspaceClient() {
  const [
    objective,
    setObjective,
  ] =
    useState(
      ""
    );


  const [
    documents,
    setDocuments,
  ] =
    useState<
      File[]
    >(
      []
    );


  const [
    datasetFiles,
    setDatasetFiles,
  ] =
    useState<
      File[]
    >(
      []
    );


  const [
    ingestion,
    setIngestion,
  ] =
    useState<
      MultiDatasetIngestion |
      null
    >(
      null
    );


  const [
    preparationSession,
    setPreparationSession,
  ] =
    useState<
      PreparationSessionView |
      null
    >(
      null
    );


  const [
    preparationSessionLoading,
    setPreparationSessionLoading,
  ] =
    useState(
      false
    );


  const [
    preparationSessionError,
    setPreparationSessionError,
  ] =
    useState<
      string |
      null
    >(
      null
    );


  const [
    activeWorkflowRestoreComplete,
    setActiveWorkflowRestoreComplete,
  ] =
    useState(
      false
    );


  const [
    qualityReport,
    setQualityReport,
  ] =
    useState<
      DataQualityReportView |
      null
    >(
      null
    );


  const [
    qualityLoading,
    setQualityLoading,
  ] =
    useState(
      false
    );


  const [
    qualityError,
    setQualityError,
  ] =
    useState<
      string |
      null
    >(
      null
    );


  const [
    cleaningPlan,
    setCleaningPlan,
  ] =
    useState<
      CleaningPlanView |
      null
    >(
      null
    );


  const [
    cleaningPlanLoading,
    setCleaningPlanLoading,
  ] =
    useState(
      false
    );


  const [
    cleaningPlanError,
    setCleaningPlanError,
  ] =
    useState<
      string |
      null
    >(
      null
    );


  const [
    selectedCleaningActionIds,
    setSelectedCleaningActionIds,
  ] =
    useState<
      string[]
    >(
      []
    );


  const [
    cleaningApplyLoading,
    setCleaningApplyLoading,
  ] =
    useState(
      false
    );


  const [
    cleaningApplyError,
    setCleaningApplyError,
  ] =
    useState<
      string |
      null
    >(
      null
    );


  const [
    preparedExportLoading,
    setPreparedExportLoading,
  ] =
    useState(
      false
    );


  const [
    preparedExportError,
    setPreparedExportError,
  ] =
    useState<
      string |
      null
    >(
      null
    );


  const [
    cleaningExecution,
    setCleaningExecution,
  ] =
    useState<
      CleaningExecutionView |
      null
    >(
      null
    );


  const [
    appliedCleaningActionIds,
    setAppliedCleaningActionIds,
  ] =
    useState<
      string[]
    >(
      []
    );




  const [
    finalValidationLoading,
    setFinalValidationLoading,
  ] =
    useState(
      false
    );


  const [
    finalValidationError,
    setFinalValidationError,
  ] =
    useState<
      string |
      null
    >(
      null
    );

                                                                                                                                const {
                                                                  semanticReview,
                                                                  setSemanticReview,
                                                                  semanticReviewLoading,
                                                                  setSemanticReviewLoading,
                                                                  semanticReviewError,
                                                                  setSemanticReviewError,
                                                                  semanticCleaningPlan,
                                                                  semanticPlanLoading,
                                                                  setSemanticPlanLoading,
                                                                  semanticPlanError,
                                                                  setSemanticPlanError,
                                                                  selectedSemanticActionIds,
                                                                  setSelectedSemanticActionIds,
                                                                  semanticCanonicalValues,
                                                                  setSemanticCanonicalValues,
                                                                  semanticApplyLoading,
                                                                  semanticApplyError,
                                                                  semanticCleaningExecution,
                                                                  appliedSemanticChoices,
                                                                  confirmedSemanticIssueIds,
                                                                  setConfirmedSemanticIssueIds,
                                                                  semanticManualResolutionNotes,
                                                                  setSemanticManualResolutionNotes,
                                                                  semanticConfirmation,
                                                                  setSemanticConfirmation,
                                                                  semanticConfirmationLoading,
                                                                  setSemanticConfirmationLoading,
                                                                  semanticConfirmationError,
                                                                  setSemanticConfirmationError,
                                                                  restoreFromPreparationUiState,
                                                                  handleSetSemanticDecision,
                                                                  handleSemanticCanonicalChange,
                                                                  handleToggleSemanticIssueConfirmation,
                                                                  handleSemanticManualResolutionChange,
                                                                  resetSemanticPreparation,
                                                                  buildSemanticCleaningPlan,
                                                                  beginSemanticReviewRun,
                                                                  prepareSemanticConfirmationRun,
                                                                  prepareSemanticCleaningApply,
                                                                  completeSemanticCleaningApply,
                                                                  failSemanticCleaningApply,
                                                                  finishSemanticCleaningApply,
                                                                } =
                                                                  useSemanticPreparationState({
      apiUrl:
        API_URL,

      clearFinalValidationError:
        () => {
          setFinalValidationError(
            null
          );
        },

      resetFinalValidationUiState:
        () => {
          setFinalValidationLoading(
            false
          );

          setFinalValidationError(
            null
          );
        },
    });


  const [
    activeDatasetIndex,
    setActiveDatasetIndex,
  ] =
    useState(
      0
    );


  const [
    ingestionLoading,
    setIngestionLoading,
  ] =
    useState(
      false
    );


  const [
    analysisLoading,
    setAnalysisLoading,
  ] =
    useState(
      false
    );


  const [
    aiPlanReport,
    setAiPlanReport,
  ] =
    useState<
      AIPlannerReportView |
      null
    >(
      null
    );


  const [
    aiPlanError,
    setAiPlanError,
  ] =
    useState<
      string |
      null
    >(
      null
    );


  const [
    aiNativeLoading,
    setAiNativeLoading,
  ] =
    useState(
      false
    );


  const [
    aiNativeReport,
    setAiNativeReport,
  ] =
    useState<
      AINativePipelineReportView |
      null
    >(
      null
    );


  const [
    aiNativeError,
    setAiNativeError,
  ] =
    useState<
      string |
      null
    >(
      null
    );


  const [
    analysisFollowUpPrompt,
    setAnalysisFollowUpPrompt,
  ] =
    useState(
      ""
    );


  const [
    analysisFollowUpLoading,
    setAnalysisFollowUpLoading,
  ] =
    useState(
      false
    );


  const [
    analysisFollowUpError,
    setAnalysisFollowUpError,
  ] =
    useState<
      string |
      null
    >(
      null
    );


  const [
    analysisFollowUpTurns,
    setAnalysisFollowUpTurns,
  ] =
    useState<
      AnalysisFollowUpTurn[]
    >(
      []
    );


  const [
    initialPromptIncludedInReport,
    setInitialPromptIncludedInReport,
  ] =
    useState(
      false
    );


  const [
    reportSelectionDetails,
    setReportSelectionDetails,
  ] =
    useState<
      ReportSelectionDetailsView |
      null
    >(
      null
    );


  const [
    reportAvailableAnalyses,
    setReportAvailableAnalyses,
  ] =
    useState<
      ReportAvailableAnalysisDetailView[]
    >(
      []
    );


  const [
    reportSelectionLoading,
    setReportSelectionLoading,
  ] =
    useState(
      false
    );


  const [
    reportSelectionError,
    setReportSelectionError,
  ] =
    useState<
      string |
      null
    >(
      null
    );


  const [
    requestedResolutionLoadingId,
    setRequestedResolutionLoadingId,
  ] =
    useState<
      string |
      null
    >(
      null
    );


  const [
    requestedResolutionErrors,
    setRequestedResolutionErrors,
  ] =
    useState<
      Record<
        string,
        string
      >
    >(
      {}
    );


  const [
    pdfExportLoading,
    setPdfExportLoading,
  ] =
    useState(
      false
    );


  const [
    report,
    setReport,
  ] =
    useState<
      RoutedUnifiedAnalysisReportView |
      null
    >(
      null
    );


  const [
    ragReport,
    setRagReport,
  ] =
    useState<
      RagContextReport |
      null
    >(
      null
    );


  const [
    documentSummary,
    setDocumentSummary,
  ] =
    useState<
      DocumentSummaryView |
      null
    >(
      null
    );


  const [
    requestedPlan,
    setRequestedPlan,
  ] =
    useState<
      RequestedPlanView |
      null
    >(
      null
    );


  const [
    error,
    setError,
  ] =
    useState<
      string |
      null
    >(
      null
    );


  const [
    activeStep,
    setActiveStep,
  ] =
    useState<
      WorkspaceStep
    >(
      "data"
    );


  const [
    activePreparationStep,
    setActivePreparationStep,
  ] =
    useState<
      PreparationSubstep
    >(
      "understand"
    );

  /*
   * PREPARATION_WORKFLOW_METADATA_WORKSPACE_V0_1
   *
   * Browser-only draft for the next workflow name.
   * Persisted workflow metadata remains server-owned.
   */
  const [
    workflowDisplayName,
    setWorkflowDisplayName,
  ] =
    useState(
      ""
    );



  useEffect(
    () => {
      const storedWorkflowId =
        readActivePreparationWorkflowId();


      if (
        !storedWorkflowId
      ) {
        setActiveWorkflowRestoreComplete(
          true
        );

        return;
      }


      const controller =
        new AbortController();


      setPreparationSessionLoading(
        true
      );

      setPreparationSessionError(
        null
      );


      void (
        async () => {
          try {
            const restoredSession =
              await getPreparationSession(
                storedWorkflowId,
                controller.signal
              );


            setPreparationSession(
              restoredSession
            );

            setActivePreparationStep(
              preparationSubstepFromSession(
                restoredSession
              )
            );




            /*
             * A persisted validated workflow already owns its
             * analytical results server-side.
             *
             * A previous Report surface returns to Analyses
             * after F5 because the transient unified `report`
             * object is intentionally not fabricated.
             */
            const restoredStep: WorkspaceStep =
              resolveRestoredWorkspaceStep(
                storedWorkflowId,
                restoredSession
                  .snapshot
                  .ready_for_analysis
              );


            setActiveStep(
              restoredStep
            );


            /*
             * Restore the frontend dataset context from
             * server-owned Preparation artifacts.
             *
             * Browser File objects are intentionally not
             * required after the initial upload.
             */
            const restoredIngestion =
              await getPreparationIngestionView(
                storedWorkflowId,
                controller.signal
              );


            if (
              controller.signal.aborted
            ) {
              return;
            }


            setIngestion(
              restoredIngestion
            );


            /*
             * PREPARATION_UI_STATE_RESTORE_V0_1
             *
             * Restore committed structured outputs only.
             *
             * No ingestion replay.
             * No quality recomputation.
             * No cleaning execution.
             * No semantic LLM call.
             */
            const restoredUiState =
              await getPreparationUiState(
                storedWorkflowId,
                controller.signal
              );


            if (
              controller.signal.aborted
            ) {
              return;
            }


            // ================================================
            // QUALITY
            // ================================================

            setQualityReport(
              restoredUiState
                .quality_report
            );


            // ================================================
            // DETERMINISTIC CLEANING
            // ================================================

            setCleaningPlan(
              restoredUiState
                .cleaning_plan
            );

            setCleaningExecution(
              restoredUiState
                .cleaning_execution
            );


            const restoredAppliedCleaningActionIds =
              deriveRestoredAppliedCleaningActionIds(
                restoredUiState
              );


            setAppliedCleaningActionIds(
              restoredAppliedCleaningActionIds
            );


            /*
             * Before execution, deterministic safe candidates
             * use the same default selection as the normal
             * cleaning-plan response.
             *
             * After execution, only actually applied actions
             * are restored as selected.
             */
            const restoredSelectedCleaningActionIds =
              deriveRestoredSelectedCleaningActionIds(
                restoredUiState,
                restoredAppliedCleaningActionIds
              );


            setSelectedCleaningActionIds(
              restoredSelectedCleaningActionIds
            );


            // ================================================
            // SEMANTIC REVIEW
            // ================================================

            restoreFromPreparationUiState(
              restoredUiState
            );
          } catch (
            caughtError
          ) {
            if (
              controller.signal.aborted
            ) {
              return;
            }


            if (
              caughtError instanceof
                PreparationApiError &&
              caughtError.status ===
                404
            ) {
              clearActivePreparationWorkflowId();

              return;
            }


            setPreparationSessionError(
              caughtError
                instanceof Error
                ? caughtError.message
                : "Impossible de restaurer le workspace actif."
            );
          } finally {
            if (
              !controller.signal.aborted
            ) {
              setPreparationSessionLoading(
                false
              );

              setActiveWorkflowRestoreComplete(
                true
              );
            }
          }
        }
      )();


      return () => {
        controller.abort();
      };
    },
    []
  );


  useEffect(
    () => {
      const workflowId =
        preparationSession
          ?.workflow_id;


      if (
        workflowId
      ) {
        persistActivePreparationWorkflowId(
          workflowId
        );
      }
    },
    [
      preparationSession
        ?.workflow_id,
    ]
  );


  useEffect(
    () => {
      if (
        !activeWorkflowRestoreComplete
      ) {
        return;
      }


      const workflowId =
        preparationSession
          ?.workflow_id;


      if (
        !workflowId
      ) {
        return;
      }


      persistActiveWorkspaceStep(
        workflowId,
        activeStep
      );
    },
    [
      activeStep,
      activeWorkflowRestoreComplete,
      preparationSession
        ?.workflow_id,
    ]
  );


  useEffect(
    () => {
      if (
        !preparationSession
      ) {
        return;
      }


      setActivePreparationStep(
        preparationSubstepFromSession(
          preparationSession
        )
      );
    },
    [
      preparationSession
        ?.workflow_id,
    ]
  );


  useEffect(
    () => {
      setAnalysisFollowUpPrompt(
        ""
      );

      setAnalysisFollowUpError(
        null
      );

      setAnalysisFollowUpTurns(
        []
      );

      setInitialPromptIncludedInReport(
        false
      );

      setReportSelectionDetails(
        null
      );

      setReportAvailableAnalyses(
        []
      );

      setReportSelectionError(
        null
      );

      setRequestedResolutionLoadingId(
        null
      );

      setRequestedResolutionErrors(
        {}
      );


      const workflowId =
        preparationSession
          ?.workflow_id;


      if (
        !workflowId
      ) {
        return;
      }


      void refreshReportSelection(
        workflowId
      );
    },
    [
      preparationSession
        ?.workflow_id,
    ]
  );


  const activeManifest:
    DatasetManifest |
    null =
      ingestion
        ?.datasets[
          activeDatasetIndex
        ] ??
      null;


  const signalKpis =
    useMemo(
      () =>
        report
          ? buildSignalKpis(
              report
            )
          : [],
      [
        report,
      ]
    );


  const ragContextByAnalysisId =
    useMemo(
      () => {
        const lookup =
          new Map<
            string,
            FindingRagContext
          >();


        for (
          const context
          of ragReport?.contexts ??
          []
        ) {
          lookup.set(
            context.analysis_id,
            context
          );
        }


        return lookup;
      },
      [
        ragReport,
      ]
    );


  const preparationReadyForAnalysis =
    Boolean(
      preparationSession
        ?.snapshot
        .ready_for_analysis
    );


  const submitDisabled =
    Boolean(
      analysisLoading ||
      ingestionLoading ||
      preparationSessionLoading ||
      !activeWorkflowRestoreComplete ||
      cleaningApplyLoading ||
      !preparationReadyForAnalysis
    );


  const latestAnalysisFollowUp =
    analysisFollowUpTurns.length >
      0
      ? analysisFollowUpTurns[
          analysisFollowUpTurns.length -
          1
        ]
      : null;


  const selectedPromptAnalyses =
    useMemo<
      ReportPromptAnalysisView[]
    >(
      () =>
        (
          reportSelectionDetails
            ?.analyses ??
          []
        ).map(
          (
            detail
          ) => ({
            id:
              detail
                .selection
                .analysis_id,

            source_type:
              detail
                .selection
                .source_type,

            source_label:
              reportSourceLabel(
                detail
                  .selection
                  .source_type
              ),

            objective:
              detail
                .selection
                .objective,

            report:
              detail
                .pipeline_payload,
          })
        ),
      [
        reportSelectionDetails,
      ]
    );



  const reportAvailableAnalysisById =
    useMemo(
      () =>
        new Map(
          reportAvailableAnalyses.map(
            (
              analysis
            ) => [
              analysis.analysis_id,
              analysis,
            ] as const
          )
        ),
      [
        reportAvailableAnalyses,
      ]
    );


  const selectedReportAnalysisIds =
    useMemo(
      () =>
        new Set(
          selectedPromptAnalyses.map(
            (
              analysis
            ) =>
              analysis.id
          )
        ),
      [
        selectedPromptAnalyses,
      ]
    );


  const unresolvedDocumentRequests =
    useMemo(
      () =>
        reportAvailableAnalyses
          .filter(
            (
              analysis
            ) =>
              analysis.source_type ===
                "document_request" &&
              !analysis.executed &&
              requestedLifecycleForAnalysis(
                analysis
              ) !==
                null
          )
          .slice()
          .sort(
            (
              left,
              right
            ) =>
              requestedLifecycleOrder(
                left
              ) -
              requestedLifecycleOrder(
                right
              )
          ),
      [
        reportAvailableAnalyses,
      ]
    );


  const unselectedRequestedAnalyses =
    useMemo(
      () =>
        reportAvailableAnalyses.filter(
          (
            analysis
          ) =>
            analysis.executed &&
            analysis.source_type !==
              "automatic" &&
            !selectedReportAnalysisIds.has(
              analysis.analysis_id
            )
        ),
      [
        reportAvailableAnalyses,
        selectedReportAnalysisIds,
      ]
    );


  const unselectedAutomaticAnalyses =
    useMemo(
      () =>
        reportAvailableAnalyses.filter(
          (
            analysis
          ) =>
            analysis.executed &&
            analysis.source_type ===
              "automatic" &&
            !selectedReportAnalysisIds.has(
              analysis.analysis_id
            )
        ),
      [
        reportAvailableAnalyses,
        selectedReportAnalysisIds,
      ]
    );


  const plannerModelForUi =
    aiNativeReport
      ?.planner_model ??
    aiPlanReport
      ?.model ??
    null;


  const activePlannerUi =
    plannerUiCopy(
      plannerModelForUi
    );


  const dataReady =
    Boolean(
      ingestion ||
      preparationSession
    );


  const reportReady =
    report !==
    null;


  const interventionCount =
    requestedPlan?.requests.filter(
      (
        request
      ) =>
        request.status !==
        "ready"
    ).length ??
    0;


  const loadCleaningPlan =
    createCleaningPlanLoader({
      apiUrl:
        API_URL,

      setPreparationSession,

      setCleaningPlan,

      setCleaningPlanLoading,

      setCleaningPlanError,

      setSelectedCleaningActionIds,

      setCleaningExecution,

      setAppliedCleaningActionIds,
    });


  function handleToggleCleaningAction(
    actionId:
      string
  ) {
    if (
      cleaningExecution
    ) {
      return;
    }


    setSelectedCleaningActionIds(
      (
        current
      ) =>
        current.includes(
          actionId
        )
          ? current.filter(
              (
                value
              ) =>
                value !==
                actionId
            )
          : [
              ...current,
              actionId,
            ]
    );

    setCleaningApplyError(
      null
    );

    resetSemanticPreparation();
  }


  async function handleApplyCleaning() {
        const preparedCleaningApply =
      prepareDeterministicCleaningApply({
        datasetFiles,
        cleaningPlan,
        preparationSession,
        selectedCleaningActionIds,
      });


    if (
      !preparedCleaningApply.ok
    ) {
      setCleaningApplyError(
        preparedCleaningApply.error
      );

      return;
    }


    setCleaningApplyLoading(
      true
    );


    setCleaningApplyError(
      null
    );


    try {
                        const typedPayload =
              await applyDeterministicCleaning({
                datasetFiles,

                workflowId:
                  preparedCleaningApply.workflowId,

                approvedCleaningActionIds:
                  preparedCleaningApply.approvedCleaningActionIds,

                rejectedCleaningActionIds:
                  preparedCleaningApply.rejectedCleaningActionIds,
              });


      setQualityReport(
        typedPayload
          .quality_report
      );

      setCleaningPlan(
        typedPayload
          .cleaning_plan
      );

      setCleaningExecution(
        typedPayload
          .execution
      );


      const appliedIds =
        typedPayload
          .execution
          .action_results
          .filter(
            (
              result
            ) =>
              result.status ===
              "applied"
          )
          .map(
            (
              result
            ) =>
              result.action_id
          );


      setAppliedCleaningActionIds(
        appliedIds
      );

      setSelectedCleaningActionIds(
        appliedIds
      );


      resetSemanticPreparation();


      const synchronizedSession =
        await getPreparationSession(
          preparedCleaningApply.workflowId
        );


      setPreparationSession(
        synchronizedSession
      );


      setActivePreparationStep(
        preparationSubstepFromSession(
          synchronizedSession
        )
      );


      setReport(
        null
      );

      setRagReport(
        null
      );

      setDocumentSummary(
        null
      );

      setRequestedPlan(
        null
      );

      setAiNativeReport(
        null
      );

      setAiNativeError(
        null
      );
    } catch (
      caughtError
    ) {
      setCleaningExecution(
        null
      );

      setAppliedCleaningActionIds(
        []
      );

      setCleaningApplyError(
        caughtError
          instanceof Error
          ? caughtError.message
          : "Le nettoyage contrôlé a échoué."
      );
    } finally {
      setCleaningApplyLoading(
        false
      );
    }
  }







  async function handleRunSemanticReview() {
    if (
      datasetFiles.length ===
      0
    ) {
      setSemanticReviewError(
        "Ajoutez au moins un fichier CSV."
      );

      return;
    }


    if (
      preparationSession ===
      null
    ) {
      setSemanticReviewError(
        "Aucune session de préparation active."
      );

      return;
    }


    const deterministicCleaningReady =
      cleaningPlan !==
        null &&
      (
        cleaningPlan.action_count ===
          0 ||
        cleaningExecution !==
          null
      );


    if (
      !deterministicCleaningReady
    ) {
      setSemanticReviewError(
        "Terminez d’abord l’étape de nettoyage déterministe."
      );

      return;
    }


        beginSemanticReviewRun();


    try {
            const typedReview =
        await requestSemanticReview({
          datasetFiles,

          workflowId:
            preparationSession.workflow_id,

          approvedCleaningActionIds:
            appliedCleaningActionIds,

          model:
            "gemma3:4b",
        });


      setSemanticReview(
        typedReview
      );


      const synchronizedSession =
        await getPreparationSession(
          preparationSession.workflow_id
        );


      setPreparationSession(
        synchronizedSession
      );


            await buildSemanticCleaningPlan({
        review:
          typedReview,

        preparationSession,
        datasetFiles,
        appliedCleaningActionIds,
      });
    } catch (
      caughtError
    ) {
      setSemanticReview(
        null
      );

      setSemanticReviewError(
        caughtError
          instanceof Error
          ? caughtError.message
          : "La revue sémantique locale a échoué."
      );
    } finally {
      setSemanticReviewLoading(
        false
      );
    }
  }






  async function handleApplySemanticCleaning() {
        const preparedSemanticApply =
      prepareSemanticCleaningApply({
        workflowId:
          preparationSession?.workflow_id ??
          null,
      });

    if (
      preparedSemanticApply ===
        null
    ) {
      return;
    }

    const {
      workflowId,
      semanticDecisions,
      choices,
    } =
      preparedSemanticApply;


    try {
                        const typedPayload =
              await applySemanticCleaning({
                datasetFiles,

                workflowId,

                approvedCleaningActionIds:
                  appliedCleaningActionIds,

                semanticDecisions,

                approvedSemanticChoices:
                  choices,
              });


            completeSemanticCleaningApply({
        response:
          typedPayload,

        choices,
      });

      setFinalValidationError(
        null
      );


      setReport(
        null
      );

      setRagReport(
        null
      );

      setDocumentSummary(
        null
      );

      setRequestedPlan(
        null
      );

      setAiNativeReport(
        null
      );

      setAiNativeError(
        null
      );
    } catch (
      caughtError
    ) {
            failSemanticCleaningApply(
        caughtError
      );
    } finally {
            finishSemanticCleaningApply();
    }
  }






  async function handleConfirmSemanticReview() {
    if (
      semanticReview ===
      null
    ) {
      setSemanticConfirmationError(
        "Aucune revue sémantique n’est disponible."
      );

      return;
    }


    if (
      preparationSession ===
      null
    ) {
      setSemanticConfirmationError(
        "Aucune session de préparation active."
      );

      return;
    }


    if (
      semanticReview.decisions.length ===
      0
    ) {
      setSemanticConfirmationError(
        "Aucune décision sémantique confirmable n’est disponible. Les problèmes protégés doivent être examinés manuellement."
      );

      return;
    }


        const manualResolutions =
      prepareSemanticConfirmationRun(
        semanticReview
      );


    try {
      const response =
        await confirmSemanticReview(
          {
            datasetFiles,

            workflowId:
              preparationSession.workflow_id,

            semanticDecisions:
              semanticReview.decisions,

            confirmedIssueIds:
              confirmedSemanticIssueIds,

            approvedSemanticChoices:
              appliedSemanticChoices,

            manualResolutions,

            approvedCleaningActionIds:
              appliedCleaningActionIds,
          }
        );


      setSemanticConfirmation(
        response.confirmation
      );


      const synchronizedSession =
        await getPreparationSession(
          preparationSession.workflow_id
        );


      setPreparationSession(
        synchronizedSession
      );


      setActivePreparationStep(
        preparationSubstepFromSession(
          synchronizedSession
        )
      );
    } catch (
      caughtError
    ) {
            if (
        caughtError instanceof
          SemanticConfirmationApiError &&
        caughtError.confirmation
      ) {
        setSemanticConfirmation(
          caughtError.confirmation
        );
      }


      setSemanticConfirmationError(
        caughtError
          instanceof Error
          ? caughtError.message
          : "La confirmation de la revue sémantique a échoué."
      );


      try {
        const synchronizedSession =
          await getPreparationSession(
            preparationSession.workflow_id
          );

        setPreparationSession(
          synchronizedSession
        );
      } catch {
        // Preserve the original confirmation error.
      }
    } finally {
      setSemanticConfirmationLoading(
        false
      );
    }
  }


  async function handleValidatePreparation() {
    if (
      preparationSession ===
      null
    ) {
      setFinalValidationError(
        "Aucune session de préparation active."
      );

      return;
    }


    setFinalValidationLoading(
      true
    );

    setFinalValidationError(
      null
    );


    try {
      const validatedSession =
        await validatePreparationSession(
          preparationSession.workflow_id
        );


      setPreparationSession(
        validatedSession
      );
    } catch (
      caughtError
    ) {
      setFinalValidationError(
        caughtError
          instanceof Error
          ? caughtError.message
          : "La validation finale de la préparation a échoué."
      );


      try {
        const synchronizedSession =
          await getPreparationSession(
            preparationSession.workflow_id
          );

        setPreparationSession(
          synchronizedSession
        );
      } catch {
        // Preserve the original validation error.
      }
    } finally {
      setFinalValidationLoading(
        false
      );
    }
  }


  async function handleRefreshPreparationSession() {
    if (
      !preparationSession
    ) {
      return;
    }


    setPreparationSessionLoading(
      true
    );

    setPreparationSessionError(
      null
    );


    try {
      const refreshedSession =
        await getPreparationSession(
          preparationSession.workflow_id
        );


      setPreparationSession(
        refreshedSession
      );
    } catch (
      caughtError
    ) {
      setPreparationSessionError(
        caughtError
          instanceof Error
          ? caughtError.message
          : "Impossible d’actualiser la session de préparation."
      );
    } finally {
      setPreparationSessionLoading(
        false
      );
    }
  }



  /*
   * PREPARATION_WORKFLOW_HISTORY_FRONTEND_V0_1
   *
   * Opening an existing server-owned workflow deliberately
   * reuses the already validated browser-refresh restoration
   * path.
   *
   * We only change the active workflow identity here.
   * The subsequent reload restores:
   *
   * - PreparationSession;
   * - Preparation ingestion view;
   * - Preparation UI state;
   * - workspace step;
   * - server-owned analyses / report selection.
   */
  function handleOpenHistoricalWorkflow(
    workflowId:
      string
  ) {
    const normalizedWorkflowId =
      workflowId.trim();


    if (
      !normalizedWorkflowId
    ) {
      return;
    }


    persistActivePreparationWorkflowId(
      normalizedWorkflowId
    );


    if (
      typeof window !==
        "undefined"
    ) {
      window.location.reload();
    }
  }


  /*
   * EXPLICIT_NEW_WORKFLOW_V0_2
   *
   * Detach only the browser from the current workflow.
   * The existing server-owned workflow is deliberately kept.
   *
   * The next successful CSV import will create another
   * PreparationSession with a new backend-generated workflow_id.
   */
    const resetPreparationPipelineState =
    createPreparationPipelineReset({
      setIngestion,
      setIngestionLoading,
      setPreparationSession,
      setPreparationSessionError,
      setPreparationSessionLoading,
      setQualityReport,
      setQualityError,
      setQualityLoading,
      setCleaningPlan,
      setCleaningPlanError,
      setCleaningPlanLoading,
      setSelectedCleaningActionIds,
      setCleaningExecution,
      setAppliedCleaningActionIds,
      setCleaningApplyError,
      setCleaningApplyLoading,
      setPreparedExportError,
      setPreparedExportLoading,
    });

    const resetAnalysisOutputs =
      createAnalysisOutputReset({
        setReport,
        setRagReport,
        setDocumentSummary,
        setRequestedPlan,
        setAiPlanReport,
        setAiPlanError,
        setAiNativeReport,
        setAiNativeError,
      });

function handleStartNewWorkflow() {
    clearActivePreparationWorkflowId();


    setWorkflowDisplayName(
      ""
    );


    // --------------------------------------------------------
    // INPUT CONTEXT
    // --------------------------------------------------------

    setDatasetFiles(
      []
    );

    setDocuments(
      []
    );

    setObjective(
      ""
    );

      resetPreparationPipelineState({
    resetIngestionLoading:
      true,

    resetQualityLoading:
      true,

    resetCleaningPlanLoading:
      true,

    resetCleaningApplyLoading:
      true,

    resetPreparedExportError:
      true,

    resetPreparedExportLoading:
      true,
  });


    // --------------------------------------------------------
    // SEMANTIC PREPARATION
    // --------------------------------------------------------

    resetSemanticPreparation();


    // --------------------------------------------------------
    // FINAL VALIDATION
    // --------------------------------------------------------

    setFinalValidationError(
      null
    );

    setFinalValidationLoading(
      false
    );


    // --------------------------------------------------------
    // ANALYTICAL OUTPUTS
    // --------------------------------------------------------

        resetAnalysisOutputs();

    setError(
      null
    );


    /*
     * Existing preparationSession.workflow_id effects also
     * clear report-selection / follow-up state after the
     * active session becomes null.
     */


    // --------------------------------------------------------
    // NAVIGATION
    // --------------------------------------------------------

    setActiveDatasetIndex(
      0
    );

    setActivePreparationStep(
      "understand"
    );

    setActiveStep(
      "data"
    );
  }


  async function handleDatasetsChange(
    event:
      ChangeEvent<HTMLInputElement>
  ) {
    const files =
      Array.from(
        event.target.files ??
        []
      );


    /*
     * Cancelling the native file picker is not a new
     * workflow decision. Preserve the current state.
     */
    if (
      files.length ===
        0
    ) {
      return;
    }


    /*
     * Defence in depth: a real new import always
     * detaches the previous browser workflow identity.
     */
    clearActivePreparationWorkflowId();


    setDatasetFiles(
      files
    );

        resetPreparationPipelineState({
      resetPreparedExportError:
        true,
    });

    resetSemanticPreparation();

        resetAnalysisOutputs();

    setError(
      null
    );

    setActiveDatasetIndex(
      0
    );


    setActiveStep(
      "data"
    );





    setIngestionLoading(
      true
    );


    try {
            const typedIngestion =
        await ingestDatasetFiles(
          files
        );


      setIngestion(
        typedIngestion
      );


      let createdPreparationSession:
        PreparationSessionView |
        null =
          null;


      setPreparationSessionLoading(
        true
      );

      setPreparationSessionError(
        null
      );


      try {
        const createdSession =
          await createPreparationSession(
            typedIngestion.datasets.map(
              (
                dataset
              ) =>
                dataset.dataset_id
            ),

            undefined,

            workflowDisplayName
          );


        createdPreparationSession =
          createdSession;


        setPreparationSession(
          createdSession
        );
      } catch (
        sessionCaughtError
      ) {
        setPreparationSession(
          null
        );

        setPreparationSessionError(
          sessionCaughtError
            instanceof Error
            ? sessionCaughtError.message
            : "Impossible de créer la session de préparation."
        );
      } finally {
        setPreparationSessionLoading(
          false
        );
      }


      setQualityLoading(
        true
      );

      setQualityError(
        null
      );


      try {
                if (
          !createdPreparationSession
        ) {
          throw new Error(
            "La session de préparation n’a pas pu être créée avant le diagnostic qualité."
          );
        }


        const qualityReport =
          await runPreparationQuality({
            files,

            workflowId:
              createdPreparationSession.workflow_id,
          });


        setQualityReport(
          qualityReport
        );


        const synchronizedSession =
          await getPreparationSession(
            createdPreparationSession.workflow_id
          );


        setPreparationSession(
          synchronizedSession
        );
      } catch (
        qualityCaughtError
      ) {
        setQualityReport(
          null
        );

        setQualityError(
          qualityCaughtError
            instanceof Error
            ? qualityCaughtError.message
            : "Impossible d’exécuter le diagnostic qualité."
        );
      } finally {
        setQualityLoading(
          false
        );
      }


      if (
        createdPreparationSession
      ) {
        await loadCleaningPlan(
          files,
          createdPreparationSession.workflow_id
        );
      }
    } catch (
      caughtError
    ) {
      setDatasetFiles(
        []
      );

            resetPreparationPipelineState({
        resetQualityLoading:
          true,
      });

      setError(
        caughtError
          instanceof Error
          ? caughtError.message
          : "Impossible de lire les fichiers."
      );
    } finally {
      setIngestionLoading(
        false
      );
    }
  }


  function handleDocumentsChange(
    event:
      ChangeEvent<HTMLInputElement>
  ) {
    const files =
      Array.from(
        event.target.files ??
        []
      );


    setDocuments(
      files
    );

    setReport(
      null
    );

    setRagReport(
      null
    );

    setDocumentSummary(
      null
    );

    setRequestedPlan(
      null
    );

    setError(
      null
    );
  }


  const handleExportPreparedData =
    createPreparedDataExportHandler({
      apiUrl:
        API_URL,

      datasetFiles,

      appliedCleaningActionIds,

      setPreparedExportLoading,

      setPreparedExportError,
    });


  async function refreshReportSelection(
    workflowId:
      string
  ): Promise<
    ReportSelectionDetailsView |
    null
  > {
    setReportSelectionLoading(
      true
    );

    setReportSelectionError(
      null
    );


    try {
            const {
        selection:
          typedSelection,

        available:
          typedAvailable,
      } =
        await loadReportSelectionState(
          workflowId
        );


      setReportSelectionDetails(
        typedSelection
      );

      setReportAvailableAnalyses(
        typedAvailable
          .analyses
      );


      const {
        restoredInitialPrompt,
        initialPromptIncludedInReport,
        restoredFollowUpTurns,
      } =
        deriveReportSelectionRestoration(
          typedSelection,
          typedAvailable
            .analyses
        );


      if (
        restoredInitialPrompt
      ) {
        setAiNativeReport(
          restoredInitialPrompt
            .pipeline_payload
        );

        setAiPlanReport(
          restoredInitialPrompt
            .pipeline_payload
            .planner
        );

        setInitialPromptIncludedInReport(
          initialPromptIncludedInReport
        );
      } else {
        setInitialPromptIncludedInReport(
          false
        );
      }


      setAnalysisFollowUpTurns(
        restoredFollowUpTurns
      );


      return typedSelection;
    } catch (
      caughtError
    ) {
      setReportSelectionDetails(
        null
      );

      setReportAvailableAnalyses(
        []
      );

      setReportSelectionError(
        caughtError
          instanceof Error
          ? caughtError.message
          : "La sélection du rapport n’a pas pu être rechargée."
      );

      return null;
    } finally {
      setReportSelectionLoading(
        false
      );
    }
  }


  const {
    handleResolveRequestedRanking,
    handleReconfigureRequestedTimeSeries,
    handleResolveRequestedTimeSeries,
  } =
    createRequestedAnalysisResolutionHandlers({
      apiUrl:
        API_URL,

      preparationSession,

      setRequestedResolutionLoadingId,

      setRequestedResolutionErrors,

      refreshReportSelection,
    });







  const setPromptAnalysisReportSelection =
    createPromptAnalysisReportSelection({
      apiUrl:
        API_URL,

      hasPreparationSession:
        preparationSession !==
        null,

      preparationWorkflowId:
        preparationSession
          ?.workflow_id ??
        "",

      aiNativeAnalysisId:
        aiNativeReport
          ?.analysis_id ??
        null,

      setInitialPromptIncludedInReport,

      setReportSelectionLoading,

      setReportSelectionError,

      refreshReportSelection,
    });


  const setAvailableAnalysisReportSelection =
    createAvailableAnalysisReportSelection({
      apiUrl:
        API_URL,

      hasPreparationSession:
        preparationSession !==
        null,

      preparationWorkflowId:
        preparationSession
          ?.workflow_id ??
        "",

      setReportSelectionLoading,

      setReportSelectionError,

      refreshReportSelection,
    });


  async function handleAiNativeRun() {
    if (
      !preparationSession
    ) {
      setAiNativeError(
        "La session de préparation est indisponible. Rechargez les données avant de poursuivre."
      );

      setActiveStep(
        "preparation"
      );

      return;
    }


    if (
      !preparationSession
        .snapshot
        .ready_for_analysis
    ) {
      setAiNativeError(
        "L’exécution analytique est verrouillée jusqu’à la validation finale de la préparation."
      );

      setActiveStep(
        "preparation"
      );

      setActivePreparationStep(
        "finalization"
      );

      return;
    }


    const normalizedObjective =
      objective.trim();


    if (
      !normalizedObjective
    ) {
      setAiNativeError(
        "Décrivez d’abord ce que vous souhaitez comprendre."
      );

      return;
    }


    const workflowId =
      preparationSession.workflow_id;


    setAiNativeLoading(
      true
    );

    setAiNativeError(
      null
    );

    setAiNativeReport(
      null
    );

    setInitialPromptIncludedInReport(
      false
    );


    setAnalysisFollowUpPrompt(
      ""
    );

    setAnalysisFollowUpError(
      null
    );

    setAnalysisFollowUpTurns(
      []
    );


    try {
            const typedPayload =
        await runAiNativeAnalysis({
          workflowId,

          objective:
            normalizedObjective,

          plannerModel:
            "gemma3:4b",

          toolModel:
            "qwen2.5:1.5b-instruct",
        });


      setAiNativeReport(
        typedPayload
      );


      const refreshedSelection =
        await refreshReportSelection(
          workflowId
        );


      const initialAnalysisId =
        typedPayload
          .analysis_id
          ?.trim();


      setInitialPromptIncludedInReport(
        Boolean(
          initialAnalysisId &&
          refreshedSelection
            ?.analyses
            .some(
              (
                detail
              ) =>
                detail
                  .selection
                  .analysis_id
                ===
                initialAnalysisId
            )
        )
      );


      setAiPlanReport(
        typedPayload.planner
      );
    } catch (
      caughtError
    ) {
      setAiNativeReport(
        null
      );

      setAiNativeError(
        caughtError
          instanceof Error
          ? caughtError.message
          : "L’analyse orchestrée a échoué."
      );
    } finally {
      setAiNativeLoading(
        false
      );
    }
  }


  async function handleAnalysisFollowUpSubmit(
    event:
      FormEvent<HTMLFormElement>
  ) {
    event.preventDefault();


    const normalizedPrompt =
      analysisFollowUpPrompt
        .trim();


    if (
      !normalizedPrompt
    ) {
      setAnalysisFollowUpError(
        "Écrivez une nouvelle question à analyser."
      );

      return;
    }


    if (
      !preparationSession
    ) {
      setAnalysisFollowUpError(
        "La session de préparation est indisponible. Rechargez les données avant de poursuivre."
      );

      setActiveStep(
        "preparation"
      );

      return;
    }


    if (
      !preparationSession
        .snapshot
        .ready_for_analysis
    ) {
      setAnalysisFollowUpError(
        "La préparation doit rester validée avant toute nouvelle analyse."
      );

      setActiveStep(
        "preparation"
      );

      setActivePreparationStep(
        "finalization"
      );

      return;
    }


    const workflowId =
      preparationSession.workflow_id;


    setAnalysisFollowUpLoading(
      true
    );

    setAnalysisFollowUpError(
      null
    );


    try {
            /*
      * The validated Preparation workflow is the analytical
      * source of truth. Browser File objects are intentionally
      * not required after VALIDATE.
      */

      const typedPayload =
        await runAiNativeAnalysis({
          workflowId,

          objective:
            normalizedPrompt,

          plannerModel:
            "gemma3:4b",

          toolModel:
            "qwen2.5:1.5b-instruct",
        });


      const turnId =
        (
          typedPayload
            .analysis_id
            ?.trim()
          ||
          typedPayload.trace_id
            ?.trim()
          ||
          `${Date.now()}-${
            analysisFollowUpTurns.length +
            1
          }`
        );


      setAnalysisFollowUpTurns(
        (
          currentTurns
        ) => [
          ...currentTurns,
          {
            id:
              turnId,

            objective:
              normalizedPrompt,

            report:
              typedPayload,

            included_in_report:
              true,
          },
        ]
      );


      await refreshReportSelection(
        workflowId
      );


      setAnalysisFollowUpPrompt(
        ""
      );
    } catch (
      caughtError
    ) {
      setAnalysisFollowUpError(
        caughtError
          instanceof Error
          ? caughtError.message
          : "La nouvelle analyse a échoué."
      );
    } finally {
      setAnalysisFollowUpLoading(
        false
      );
    }
  }


  async function toggleFollowUpReportSelection(
    turnId:
      string
  ) {
    const turn =
      analysisFollowUpTurns
        .find(
          (
            currentTurn
          ) =>
            currentTurn.id ===
            turnId
        );


    if (
      !turn
    ) {
      return;
    }


    await setPromptAnalysisReportSelection(
      {
        report:
          turn.report,

        included:
          !turn.included_in_report,
      }
    );
  }


  async function removePromptAnalysisFromReport(
    analysisId:
      string
  ) {
    const selected =
      reportSelectionDetails
        ?.analyses
        .find(
          (
            detail
          ) =>
            detail
              .selection
              .analysis_id
            ===
            analysisId
        );


    if (
      !selected
    ) {
      return;
    }


    await setPromptAnalysisReportSelection(
      {
        report:
          selected
            .pipeline_payload,

        included:
          false,
      }
    );
  }


  async function handleSubmit(
    event:
      FormEvent<HTMLFormElement>
  ) {
    event.preventDefault();


    if (
      !preparationSession
    ) {
      setError(
        "La session de préparation n’est pas disponible. Rechargez les données pour en créer une nouvelle."
      );

      setActiveStep(
        "preparation"
      );

      return;
    }


    if (
      !preparationSession
        .snapshot
        .ready_for_analysis
    ) {
      setError(
        "La préparation doit être validée par le backend avant de lancer l’analyse."
      );

      setActiveStep(
        "preparation"
      );

      return;
    }


    setAnalysisLoading(
      true
    );

    setError(
      null
    );

        resetAnalysisOutputs();

    setAnalysisFollowUpPrompt(
      ""
    );

    setAnalysisFollowUpError(
      null
    );

    setAnalysisFollowUpTurns(
      []
    );

    setInitialPromptIncludedInReport(
      false
    );


    try {
            const {
        contextualized,
        payload,
      } =
        await runAnalysisRequest({
          workflowId:
            preparationSession.workflow_id,

          objective,
          documents,
        });


      if (
        contextualized
      ) {
        const contextualizedPayload =
          payload as
            RoutedContextualizedAnalysisResponseView;


        setReport(
          contextualizedPayload
            .analysis
        );

        setRagReport(
          contextualizedPayload
            .rag
        );


        const contextualizedDetails =
          contextualizedPayload as
            RoutedContextualizedAnalysisResponseView &
            {
              document_summary?:
                unknown;

              requested_analysis_plan?:
                unknown;
            };


        setDocumentSummary(
          (
            contextualizedDetails
              .document_summary ??
            null
          ) as
            DocumentSummaryView |
            null
        );

        setRequestedPlan(
          (
            contextualizedDetails
              .requested_analysis_plan ??
            null
          ) as
            RequestedPlanView |
            null
        );
      } else {
        setReport(
          payload as
            RoutedUnifiedAnalysisReportView
        );

        setRagReport(
          null
        );

        setDocumentSummary(
          null
        );

        setRequestedPlan(
          null
        );
      }


      await refreshReportSelection(
        preparationSession.workflow_id
      );


      if (
        objective.trim()
      ) {
        await handleAiNativeRun();
      }


      setActiveStep(
        "analyses"
      );
    } catch (
      caughtError
    ) {
      setError(
        caughtError
          instanceof Error
          ? caughtError.message
          : "L’analyse a échoué."
      );
    } finally {
      setAnalysisLoading(
        false
      );
    }
  }


  const handlePdfExport =
    createReportPdfExportHandler({
      apiUrl:
        API_URL,

      hasPreparationSession:
        preparationSession !==
        null,

      workflowId:
        preparationSession
          ?.workflow_id ??
        null,

      selectedCount:
        reportSelectionDetails
          ?.selected_count ??
        0,

      setPdfExportLoading,

      setError,
    });


  return (
    <main
      className={
        styles.page
      }
    >
      <div
        className={
          styles.ambient
        }
        aria-hidden="true"
      />


      <header
        className={
          styles.header
        }
      >
        <div
          className={
            styles.brand
          }
        >
          <span
            className={
              styles.brandMark
            }
            aria-hidden="true"
          />

          <strong>
            DataLens
          </strong>
        </div>


        <div
          className={
            styles.privacyStatus
          }
        >
          <span
            className={
              styles.statusDot
            }
            aria-hidden="true"
          />

          Traitement local
          {" · "}
          données privées
        </div>
      </header>


      <div
        className={
          styles.shell
        }
      >
        <section
          className={
            styles.hero
          }
        >
          <h1>
            Analysez vos données

            <span>
              avec plus de clarté.
            </span>
          </h1>

          <p>
            Décrivez ce que vous souhaitez comprendre,
            ajoutez vos données et, si nécessaire, votre
            contexte métier. DataLens confie les calculs à
            Python et utilise l’IA locale pour comprendre la
            demande, préparer le plan et contextualiser les résultats.
          </p>
        </section>


        <WorkspaceNavigation
          activeStep={
            activeStep
          }
          onStepChange={
            setActiveStep
          }
          dataReady={
            dataReady
          }
          reportReady={
            reportReady
          }
          interventionCount={
            interventionCount
          }
        />


        <form
          className={
            styles.workspace
          }
          style={{
            display:
              (
                activeStep ===
                  "analyses" ||
                activeStep ===
                  "report"
              )
                ? "none"
                : undefined,
          }}
          onSubmit={
            handleSubmit
          }
          autoComplete="off"
        >
          <section
            className={
              `${styles.panel} ${styles.objectivePanel} ${styles.analysisRequestPanel}`
            }
            style={{
              display:
                activeStep ===
                  "documents"
                  ? undefined
                  : "none",
            }}
          >
            <div
              className={
                styles.sectionHead
              }
            >
              <div>
                <span
                  className={
                    styles.eyebrow
                  }
                >
                  02 · Documents
                </span>

                <h2>
                  Votre demande d’analyse
                </h2>

                <p
                  className={
                    styles.sectionDescription
                  }
                >
                  Indiquez ce que vous voulez comprendre ou rechercher.
                  Cette demande sera traitée en priorité dans les résultats.
                </p>
              </div>

              <span
                className={
                  styles.priorityBadge
                }
              >
                Prioritaire
              </span>
            </div>


            <label
              className={
                styles.fieldLabel
              }
              htmlFor="objective"
            >
              Que voulez-vous comprendre ou rechercher ?
            </label>


            <textarea
              id="objective"
              className={
                styles.objectiveInput
              }
              value={
                objective
              }
              onChange={
                (
                  event
                ) => {
                  setObjective(
                    event.target.value
                  );

                  setAiPlanReport(
                    null
                  );

                  setAiPlanError(
                    null
                  );

                  setAiNativeReport(
                    null
                  );

                  setAiNativeError(
                    null
                  );
                }
              }
              placeholder="Ex. Valeurs atypiques · Comparer des groupes · Relation entre variables"
            />


            <p
              className={
                styles.helper
              }
            >
              Facultatif. Si vous laissez ce champ vide, DataLens explore les
              analyses compatibles automatiquement. Si vous formulez une demande,
              elle devient prioritaire ; Python conserve toujours la validation
              des datasets, des colonnes et des calculs.
            </p>
          </section>


          <div
            className={
              `${styles.sourceGrid} ${styles.sourceGridSingle}`
            }
            style={{
              display:
                (
                  activeStep ===
                    "data" ||
                  activeStep ===
                    "documents"
                )
                  ? undefined
                  : "none",
            }}
          >
            <BusinessDocumentsSection
  activeStep={activeStep}
  documents={documents}
  handleDocumentsChange={handleDocumentsChange}
/>


            <section
              className={
                `${styles.panel} ${styles.dataUploadPanel}`
              }
              style={{
                display:
                  activeStep ===
                    "data"
                    ? undefined
                    : "none",
              }}
            >
              <div
                className={
                  styles.sectionHead
                }
              >
                <div>
                  <span
                    className={
                      styles.eyebrow
                    }
                  >
                    01 · Données
                  </span>

                  <h2>
                    Importer vos données
                  </h2>

                  <p
                    className={
                      styles.sectionDescription
                    }
                  >
                    Chargez un ou plusieurs fichiers CSV. DataLens inspecte ensuite
                    leur structure avant toute préparation ou analyse.
                  </p>
                </div>

                {
                  ingestion
                    ? (
                        <span
                          className={
                            styles.sectionStatus
                          }
                        >
                          {
                            ingestion.dataset_count
                          }
                          {" fichier"}
                          {
                            ingestion.dataset_count >
                              1
                              ? "s"
                              : ""
                          }
                        </span>
                      )
                    : null
                }
              </div>


              <PreparationWorkflowHistoryPanel
                activeWorkflowId={
                  preparationSession
                    ?.workflow_id ??
                  null
                }

                activeWorkflowCanArchive={
                  Boolean(
                    preparationSession !==
                      null &&
                    ingestion !==
                      null &&
                    ingestion.dataset_count >
                      0 &&
                    !(
                      ingestionLoading ||
                      preparationSessionLoading ||
                      qualityLoading ||
                      cleaningPlanLoading ||
                      cleaningApplyLoading ||
                      semanticReviewLoading ||
                      semanticPlanLoading ||
                      semanticApplyLoading ||
                      semanticConfirmationLoading ||
                      finalValidationLoading ||
                      analysisLoading ||
                      aiNativeLoading ||
                      analysisFollowUpLoading ||
                      reportSelectionLoading ||
                      pdfExportLoading
                    )
                  )
                }

                onOpenWorkflow={
                  handleOpenHistoricalWorkflow
                }

                onArchiveActiveWorkflow={
                  handleStartNewWorkflow
                }
              />


              {
                preparationSession
                  ? (
                      <div
                        className={
                          styles.submitArea
                        }
                        style={{
                          marginTop:
                            "24px",
                        }}
                      >
                        <div
                          className={
                            styles.submitInfo
                          }
                        >
                          <strong>
                            Workflow actif
                          </strong>

                          <span>
                            {
                              ingestion
                                ? `${ingestion.dataset_count} fichier${
                                    ingestion.dataset_count >
                                      1
                                      ? "s"
                                      : ""
                                  } rattaché${
                                    ingestion.dataset_count >
                                      1
                                      ? "s"
                                      : ""
                                  } à cette préparation.`
                                : "Une préparation est actuellement active."
                            }
                            {
                              " Démarrez un nouveau workflow pour analyser d’autres données."
                            }
                          </span>
                        </div>

                        <button
                          className={
                            styles.submitButton
                          }
                          type="button"
                          onClick={
                            handleStartNewWorkflow
                          }
                          disabled={
                            ingestionLoading ||
                            preparationSessionLoading ||
                            qualityLoading ||
                            cleaningPlanLoading ||
                            cleaningApplyLoading ||
                            semanticReviewLoading ||
                            semanticPlanLoading ||
                            semanticApplyLoading ||
                            semanticConfirmationLoading ||
                            finalValidationLoading ||
                            analysisLoading ||
                            aiNativeLoading ||
                            analysisFollowUpLoading ||
                            reportSelectionLoading ||
                            pdfExportLoading
                          }
                        >
                          Nouveau workflow
                        </button>
                      </div>
                    )
                  : (
              <DatasetImportControls
  workflowDisplayName={workflowDisplayName}
  setWorkflowDisplayName={setWorkflowDisplayName}
  ingestion={ingestion}
  ingestionLoading={ingestionLoading}
  preparationSessionLoading={preparationSessionLoading}
  handleDatasetsChange={handleDatasetsChange}
/>
                    )
              }


              {
                ingestionLoading
                  ? (
                      <div
                        className={
                          styles.ingestionState
                        }
                      >
                        Lecture des fichiers…
                      </div>
                    )
                  : null
              }


              {
                ingestion
                  ? (
                      <div
                        className={
                          styles.ingestionSummary
                        }
                      >
                        <div>
                          <strong>
                            {
                              ingestion.dataset_count
                            }
                          </strong>

                          <span>
                            fichier
                            {
                              ingestion.dataset_count >
                              1
                                ? "s"
                                : ""
                            }
                          </span>
                        </div>

                        <div>
                          <strong>
                            {
                              formatNumber(
                                ingestion.total_rows
                              )
                            }
                          </strong>

                          <span>
                            lignes au total
                          </span>
                        </div>
                      </div>
                    )
                  : null
              }
            </section>
          </div>


          {
            ingestion &&
            activeStep ===
              "data"
              ? (
                  <DatasetWorkspaceSection
  ingestion={ingestion}
  activeManifest={activeManifest}
  activeDatasetIndex={activeDatasetIndex}
  setActiveDatasetIndex={setActiveDatasetIndex}
/>
                )
              : null
          }


          {
            activeStep ===
              "data" &&
            dataReady
              ? (
                  <div
                    className={
                      styles.submitArea
                    }
                  >
                    <div
                      className={
                        styles.submitInfo
                      }
                    >
                      <strong>
                        Étape Données terminée
                      </strong>

                      <span>
                        Vos fichiers sont chargés. Ajoutez maintenant une demande
                        d’analyse ou du contexte métier si nécessaire.
                      </span>
                    </div>

                    <button
                      className={
                        styles.submitButton
                      }
                      type="button"
                      onClick={
                        () =>
                          setActiveStep(
                            "documents"
                          )
                      }
                    >
                      Continuer vers Documents
                    </button>
                  </div>
                )
              : null
          }


          {
            activeStep ===
              "documents"
              ? (
                  <div
                    className={
                      styles.submitArea
                    }
                  >
                    <div
                      className={
                        styles.submitInfo
                      }
                    >
                      <strong>
                        {
                          documents.length >
                            0
                            ? `${documents.length} document${
                                documents.length >
                                1
                                  ? "s"
                                  : ""
                              } ajouté${
                                documents.length >
                                1
                                  ? "s"
                                  : ""
                              }`
                            : "Aucun document ajouté"
                        }
                      </strong>

                      <span>
                        Votre demande et le contexte éventuel sont prêts.
                        Vous pouvez poursuivre même sans document métier.
                      </span>
                    </div>

                    <button
                      className={
                        styles.submitButton
                      }
                      type="button"
                      disabled={
                        !dataReady
                      }
                      onClick={
                        () =>
                          setActiveStep(
                            "preparation"
                          )
                      }
                    >
                      Continuer vers Préparation
                    </button>
                  </div>
                )
              : null
          }


          {
            activeStep ===
              "preparation"
              ? (
                  <section
                    className={
                      styles.panel
                    }
                  >
                    <div
                      className={
                        styles.sectionHead
                      }
                    >
                      <div>
                        <span
                          className={
                            styles.eyebrow
                          }
                        >
                          Préflight
                        </span>

                        <h2>
                          Préparation avant analyse
                        </h2>

                        <p
                          className={
                            styles.resultSubtitle
                          }
                        >
                          DataLens inspecte les fichiers,
                          mesure les problèmes de qualité et
                          prépare uniquement les corrections
                          qui pourront être justifiées et tracées.
                        </p>
                      </div>
                    </div>


                    <PreparationSubstepNavigation
                      session={
                        preparationSession
                      }
                      activeStep={
                        activePreparationStep
                      }
                      onStepChange={
                        setActivePreparationStep
                      }
                      qualityReady={
                        qualityReport !==
                        null
                      }
                      cleaningPlanReady={
                        cleaningPlan !==
                        null
                      }
                      cleaningActionCount={
                        cleaningPlan
                          ?.action_count ??
                        0
                      }
                      cleaningApplied={
                        cleaningExecution !==
                        null
                      }
                      semanticReviewReady={
                        semanticReview !==
                        null
                      }
                      semanticReviewExpectedCount={
                        qualityReport
                          ?.semantic_review_count ??
                        0
                      }
                      semanticDecisionCount={
                        semanticReview
                          ?.decisions
                          .length ??
                        0
                      }
                      semanticConfirmed={
                        semanticConfirmation
                          ?.confirmed ===
                        true
                      }
                    />


                    <details
                      style={{
                        marginTop:
                          "10px",

                        padding:
                          "10px 12px",

                        border:
                          "1px solid rgba(255,255,255,0.055)",

                        borderRadius:
                          "11px",

                        background:
                          "rgba(255,255,255,0.008)",
                      }}
                    >
                      <summary
                        style={{
                          cursor:
                            "pointer",

                          fontSize:
                            "0.6rem",

                          fontWeight:
                            700,

                          opacity:
                            0.58,
                        }}
                      >
                        Voir le workflow technique
                        {" · "}
                        7 étapes backend
                      </summary>

                      <div
                        style={{
                          marginTop:
                            "10px",
                        }}
                      >
                        <PreparationWorkflowPanel
                          session={
                            preparationSession
                          }
                          loading={
                            preparationSessionLoading
                          }
                          error={
                            preparationSessionError
                          }
                          onRefresh={
                            preparationSession
                              ? handleRefreshPreparationSession
                              : undefined
                          }
                        />
                      </div>
                    </details>


                    {
                      activePreparationStep ===
                        "understand"
                        ? (
                            <PreparationUnderstandingPanel
                              ingestion={
                                ingestion
                              }
                            />
                          )
                        : null
                    }


                    {
                      activePreparationStep ===
                        "quality"
                        ? (
                            <DataPreparationStudio
                              ingestion={
                                ingestion
                              }
                              qualityReport={
                                qualityReport
                              }
                              qualityLoading={
                                qualityLoading
                              }
                              qualityError={
                                qualityError
                              }
                            />
                          )
                        : null
                    }


                    {
                      activePreparationStep ===
                        "cleaning"
                        ? (
                            <CleaningPlanPanel
                              plan={
                                cleaningPlan
                              }
                              loading={
                                cleaningPlanLoading
                              }
                              error={
                                cleaningPlanError
                              }
                              selectedActionIds={
                                selectedCleaningActionIds
                              }
                              execution={
                                cleaningExecution
                              }
                              applyLoading={
                                cleaningApplyLoading
                              }
                              applyError={
                                cleaningApplyError
                              }
                              exportLoading={
                                preparedExportLoading
                              }
                              exportError={
                                preparedExportError
                              }
                              onToggleAction={
                                handleToggleCleaningAction
                              }
                              onApply={
                                handleApplyCleaning
                              }
                              onExportPrepared={
                                handleExportPreparedData
                              }
                              onContinueSemantic={
                                () => {
                                  if (
                                    (
                                      cleaningPlan
                                        ?.protected_issue_count ??
                                      0
                                    ) >
                                    0
                                  ) {
                                    return;
                                  }


                                  setActivePreparationStep(
                                    preparationSubstepFromSession(
                                      preparationSession
                                    )
                                  );
                                }
                              }
                            />
                          )
                        : null
                    }


                    {
                      activePreparationStep ===
                        "cleaning"
                        ? (
                            <>
                              <SemanticReviewPanel
                                deterministicCleaningReady={
                                  cleaningPlan !==
                                    null &&
                                  (
                                    cleaningPlan.action_count ===
                                      0 ||
                                    cleaningExecution !==
                                      null
                                  )
                                }
                                review={
                                  semanticReview
                                }
                                reviewLoading={
                                  semanticReviewLoading
                                }
                                reviewError={
                                  semanticReviewError
                                }
                                plan={
                                  semanticCleaningPlan
                                }
                                planLoading={
                                  semanticPlanLoading
                                }
                                planError={
                                  semanticPlanError
                                }
                                selectedActionIds={
                                  selectedSemanticActionIds
                                }
                                canonicalValues={
                                  semanticCanonicalValues
                                }
                                execution={
                                  semanticCleaningExecution
                                }
                                applyLoading={
                                  semanticApplyLoading
                                }
                                applyError={
                                  semanticApplyError
                                }
                                onRunReview={
                                  handleRunSemanticReview
                                }
                                onSetDecision={
                                  handleSetSemanticDecision
                                }
                                onCanonicalChange={
                                  handleSemanticCanonicalChange
                                }
                                onApply={
                                  handleApplySemanticCleaning
                                }
                              />


                              <SemanticConfirmationPanel
                                review={
                                  semanticReview
                                }
                                plan={
                                  semanticCleaningPlan
                                }
                                execution={
                                  semanticCleaningExecution
                                }
                                confirmedIssueIds={
                                  confirmedSemanticIssueIds
                                }
                                manualResolutionNotes={
                                  semanticManualResolutionNotes
                                }
                                confirmation={
                                  semanticConfirmation
                                }
                                loading={
                                  semanticConfirmationLoading
                                }
                                error={
                                  semanticConfirmationError
                                }
                                onToggleIssue={
                                  handleToggleSemanticIssueConfirmation
                                }
                                onManualResolutionChange={
                                  handleSemanticManualResolutionChange
                                }
                                onConfirm={
                                  handleConfirmSemanticReview
                                }
                              />
                            </>
                          )
                        : null
                    }


                    {
                      activePreparationStep ===
                        "transform"
                        ? (
                            preparationStageResolved(
                              findPreparationStage(
                                preparationSession,
                                "transform"
                              )
                            )
                              ? (
                                  <PreparationResolvedStagePanel
                                    eyebrow="04 · Transformer"
                                    title={
                                      findPreparationStage(
                                        preparationSession,
                                        "transform"
                                      )?.status ===
                                        "skipped"
                                        ? "Aucune transformation requise"
                                        : "Transformations terminées"
                                    }
                                    description={
                                      findPreparationStage(
                                        preparationSession,
                                        "transform"
                                      )?.status ===
                                        "skipped"
                                        ? "Le moteur a validé qu’aucune transformation structurelle supplémentaire n’était nécessaire pour poursuivre."
                                        : "Les transformations approuvées ont été exécutées et post-validées. Les artifacts transformés restent traçables côté serveur."
                                    }
                                    skipped={
                                      findPreparationStage(
                                        preparationSession,
                                        "transform"
                                      )?.status ===
                                      "skipped"
                                    }
                                  />
                                )
                              : (
                                  <PreparationTransformPanel
                                    session={
                                      preparationSession
                                    }
                                    onSessionChange={
                                      (
                                        updatedSession
                                      ) => {
                                        setPreparationSession(
                                          updatedSession
                                        );

                                        setActivePreparationStep(
                                          preparationSubstepFromSession(
                                            updatedSession
                                          )
                                        );
                                      }
                                    }
                                  />
                                )
                          )
                        : null
                    }


                    {
                      activePreparationStep ===
                        "combine"
                        ? (
                            preparationStageResolved(
                              findPreparationStage(
                                preparationSession,
                                "combine"
                              )
                            ) &&
                            !requiresCombineDiscoveryBeforeValidation(
                              preparationSession
                            )
                              ? (
                                  <PreparationResolvedStagePanel
                                    eyebrow="05 · Assembler"
                                    title={
                                      findPreparationStage(
                                        preparationSession,
                                        "combine"
                                      )?.status ===
                                        "skipped"
                                        ? "Aucun assemblage nécessaire"
                                        : "Assemblage terminé"
                                    }
                                    description={
                                      findPreparationStage(
                                        preparationSession,
                                        "combine"
                                      )?.status ===
                                        "skipped"
                                        ? "Les données nécessaires sont déjà réunies. Aucun assemblage supplémentaire n’est nécessaire."
                                        : "Les tables ont été assemblées avec succès. Les clés utilisées et l’impact sur les lignes ont été contrôlés avant de poursuivre."
                                    }
                                    skipped={
                                      findPreparationStage(
                                        preparationSession,
                                        "combine"
                                      )?.status ===
                                      "skipped"
                                    }
                                  />
                                )
                              : (
                                  <PreparationCombinePanel
                                    session={
                                      preparationSession
                                    }
                                    onSessionChange={
                                      (
                                        updatedSession
                                      ) => {
                                        setPreparationSession(
                                          updatedSession
                                        );

                                        setActivePreparationStep(
                                          preparationSubstepFromSession(
                                            updatedSession
                                          )
                                        );
                                      }
                                    }
                                  />
                                )
                          )
                        : null
                    }


                    {
                      activePreparationStep ===
                        "finalization"
                        ? (
                            <>
                              <PreparationFinalizationPanel
                                session={
                                  preparationSession
                                }
                                loading={
                                  finalValidationLoading
                                }
                                error={
                                  finalValidationError
                                }
                                onValidate={
                                  handleValidatePreparation
                                }
                              />


                              <div
                                className={
                                  styles.metricGrid
                                }
                                style={{
                                  marginTop:
                                    "18px",
                                }}
                              >
                                <article
                                  className={
                                    styles.metricCard
                                  }
                                >
                                  <span>
                                    Sources importées
                                  </span>

                                  <strong>
                                    {
                                      ingestion?.dataset_count ??
                                      0
                                    }
                                  </strong>
                                </article>

                                <article
                                  className={
                                    styles.metricCard
                                  }
                                >
                                  <span>
                                    Lignes sources inspectées
                                  </span>

                                  <strong>
                                    {
                                      formatNumber(
                                        ingestion?.total_rows ??
                                        0
                                      )
                                    }
                                  </strong>
                                </article>

                                <article
                                  className={
                                    styles.metricCard
                                  }
                                >
                                  <span>
                                    Documents
                                  </span>

                                  <strong>
                                    {
                                      documents.length
                                    }
                                  </strong>
                                </article>

                                <article
                                  className={
                                    styles.metricCard
                                  }
                                >
                                  <span>
                                    Mode
                                  </span>

                                  <strong>
                                    {
                                      documents.length >
                                        0
                                        ? "Analyse + contexte"
                                        : "Analyse"
                                    }
                                  </strong>
                                </article>
                              </div>


                              <div
                                className={
                                  styles.summaryPanel
                                }
                              >
                                <div
                                  className={
                                    styles.summaryItem
                                  }
                                >
                                  <span>
                                    Règles de préparation
                                  </span>

                                  <p>
                                    Les jointures automatiques ne sont
                                    acceptées que si elles préservent
                                    le grain de la table de faits et
                                    satisfont les garde-fous du moteur.
                                  </p>

                                  <p>
                                    Les agrégations et variables dérivées
                                    restent déterministes et traçables.
                                    Le LLM n’est pas utilisé pour calculer
                                    les résultats statistiques.
                                  </p>
                                </div>
                              </div>
                            </>
                          )
                        : null
                    }


                  </section>
                )
              : null
          }


          <div
            className={
              styles.submitArea
            }
            style={{
              display:
                (
                  activeStep ===
                    "preparation" &&
                  activePreparationStep ===
                    "finalization"
                )
                  ? undefined
                  : "none",
            }}
          >
            <div
              className={
                styles.submitInfo
              }
            >
              {
                ingestion
                  ? (
                      <>
                        <strong>
                          {
                            ingestion.dataset_count
                          } dataset
                          {
                            ingestion.dataset_count >
                            1
                              ? "s"
                              : ""
                          }
                          {
                            documents.length >
                            0
                              ? ` · ${documents.length} document${
                                  documents.length >
                                  1
                                    ? "s"
                                    : ""
                                }`
                              : ""
                          }
                        </strong>

                        <span>
                          {
                            documents.length >
                            0
                              ? "Analyse déterministe + contexte documentaire local"
                              : "Analyse déterministe"
                          }
                        </span>
                      </>
                    )
                  : (
                      <span>
                        Ajoutez des données
                        pour commencer.
                      </span>
                    )
              }
            </div>


            <button
              className={
                styles.submitButton
              }
              type="submit"
              disabled={
                submitDisabled
              }
            >
              {
                analysisLoading
                  ? (
                      documents.length >
                      0
                        ? "Analyse et contextualisation…"
                        : "Analyse en cours…"
                    )
                  : (
                      !preparationSession
                        ?.snapshot
                        .ready_for_analysis
                        ? "Préparation à terminer"
                        : (
                            documents.length >
                            0
                              ? "Analyser avec le contexte"
                              : "Analyser les données"
                          )
                    )
              }
            </button>
          </div>


          {
            error
              ? (
                  <div
                    className={
                      styles.error
                    }
                    role="alert"
                  >
                    <strong>
                      Impossible de lancer
                      l’analyse
                    </strong>

                    <span>
                      {
                        error
                      }
                    </span>
                  </div>
                )
              : null
          }
        </form>


        {
          activeStep ===
            "analyses"
            ? (
        <AnalysisExecutionPanel
          activePlannerUi={
            activePlannerUi
          }
          aiNativeError={
            aiNativeError
          }
          aiNativeLoading={
            aiNativeLoading
          }
          aiNativeReport={
            aiNativeReport
          }
          aiPlanError={
            aiPlanError
          }
          aiPlanReport={
            aiPlanReport
          }
          objective={
            objective
          }
          preparationReadyForAnalysis={
            preparationReadyForAnalysis
          }
        />
              )
            : null
        }

        {
          activeStep ===
            "analyses" &&
          report ===
            null
            ? (
                <section
                  className={
                    styles.results
                  }
                >
                  <header
                    className={
                      styles.resultHeader
                    }
                  >
                    <div>
                      <span
                        className={
                          styles.eyebrow
                        }
                      >
                        État serveur restauré
                      </span>

                      <h2>
                        Analyses restaurées
                      </h2>

                      <p
                        className={
                          styles.resultSubtitle
                        }
                      >
                        {
                          reportAvailableAnalyses
                            .filter(
                              (
                                analysis
                              ) =>
                                analysis.executed
                            )
                            .length
                        } analyses persistées
                        {" ? "}
                        {
                          selectedReportAnalysisIds
                            .size
                        } sélectionnées pour le rapport
                      </p>
                    </div>


                    <div
                      className={
                        styles.resultMeta
                      }
                    >
                      <span>
                        Source
                      </span>

                      <strong>
                        Serveur DataLens
                      </strong>
                    </div>
                  </header>


                  {
                    reportSelectionLoading &&
                    reportAvailableAnalyses
                      .length ===
                      0
                      ? (
                          <div
                            className={
                              styles.technicalReasons
                            }
                          >
                            <p>
                              Synchronisation des analyses
                              persistées…
                            </p>
                          </div>
                        )
                      : null
                  }


                  {
                    reportSelectionError
                      ? (
                          <div
                            className={
                              styles.technicalReasons
                            }
                          >
                            <p>
                              {
                                reportSelectionError
                              }
                            </p>
                          </div>
                        )
                      : null
                  }


                  {
                    !reportSelectionLoading &&
                    reportAvailableAnalyses
                      .filter(
                        (
                          analysis
                        ) =>
                          analysis.executed
                      )
                      .length ===
                      0
                      ? (
                          <div
                            className={
                              styles.technicalReasons
                            }
                          >
                            <p>
                              Aucune analyse exécutée
                              n’est persistée pour ce workflow.
                            </p>
                          </div>
                        )
                      : null
                  }


                  <div
                    style={{
                      display:
                        "grid",

                      gap:
                        "14px",

                      marginTop:
                        "16px",
                    }}
                  >
                    {
                      reportAvailableAnalyses
                        .filter(
                          (
                            analysis
                          ) =>
                            analysis.executed
                        )
                        .map(
                          (
                            analysis,
                            index
                          ) => {
                            const selected =
                              selectedReportAnalysisIds
                                .has(
                                  analysis
                                    .analysis_id
                                );


                            const requestedFinding =
                              analysis
                                .source_type ===
                                  "document_request"
                                ? requestedFindingFromAvailableAnalysis(
                                    analysis
                                  )
                                : null;


                            return (
                              <article
                                key={
                                  analysis
                                    .analysis_id
                                }
                                style={{
                                  padding:
                                    "16px",

                                  border:
                                    selected
                                      ? "1px solid rgba(122, 203, 160, 0.20)"
                                      : "1px solid rgba(126, 177, 255, 0.14)",

                                  borderRadius:
                                    "14px",

                                  background:
                                    selected
                                      ? "rgba(4, 14, 19, 0.34)"
                                      : "rgba(3, 8, 17, 0.30)",
                                }}
                              >
                                <AvailableAnalysisSelectionHeader
  analysis={analysis}
  selected={selected}
  reportSelectionLoading={reportSelectionLoading}
  setAvailableAnalysisReportSelection={setAvailableAnalysisReportSelection}
/>


                                {
                                  requestedFinding
                                    ? (
                                        <RequestedFindingCard
                                          finding={
                                            requestedFinding
                                          }
                                          index={
                                            index
                                          }
                                          ragContext={
                                            ragContextByAnalysisId
                                              .get(
                                                requestedFinding
                                                  .analysis_id
                                              ) ??
                                            null
                                          }
                                          reconfigurationAnalysis={
                                            analysis
                                          }
                                          reconfigurationLoading={
                                            requestedResolutionLoadingId ===
                                            analysis
                                              .analysis_id
                                          }
                                          reconfigurationError={
                                            requestedResolutionErrors[
                                              analysis
                                                .analysis_id
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
                                            analysis
                                              .pipeline_payload
                                          }
                                          objective={
                                            analysis
                                              .objective
                                          }
                                        />
                                      )
                                }
                              </article>
                            );
                          }
                        )
                    }
                  </div>
                </section>
              )
            : null
        }


        {
          report &&
          (
            activeStep ===
              "analyses" ||
            activeStep ===
              "report"
          )
            ? (
                <section
                  className={
                    styles.results
                  }
                >
                  <header
                    className={
                      styles.resultHeader
                    }
                  >
                    <div>
                      <span
                        className={
                          styles.eyebrow
                        }
                      >
                        Analyse terminée
                      </span>

                      <h2
                        className={
                          styles.resultTitle
                        }
                      >
                        {
                          report.title
                        }
                      </h2>

                      <p
                        className={
                          styles.resultSubtitle
                        }
                      >
                        {
                          report.inventory.dataset_count
                        } fichiers
                        {" · "}
                        {
                          report.inventory
                            .discovered_analysis_count
                        } analyses découvertes
                        {" · "}
                        {
                          report.inventory
                            .executed_analysis_count
                        } exécutées
                      </p>
                    </div>


                    <div
                      className={
                        styles.resultMeta
                      }
                    >
                      <span>
                        Mode
                      </span>

                      <strong>
                        {
                          ragReport
                            ? "Analyse + RAG"
                            : "Analyse"
                        }
                      </strong>
                    </div>
                  </header>


                  {
                    !(
                      activeStep ===
                        "analyses" &&
                      report.entity_outlier_finding
                    )
                      ? (
                          <div
                    className={
                      styles.metricGrid
                    }
                  >
                    <article
                      className={
                        styles.metricCard
                      }
                    >
                      <span>
                        Fichiers
                      </span>

                      <strong>
                        {
                          report.inventory.dataset_count
                        }
                      </strong>
                    </article>


                    <article
                      className={
                        styles.metricCard
                      }
                    >
                      <span>
                        Analyses découvertes
                      </span>

                      <strong>
                        {
                          report.inventory
                            .discovered_analysis_count
                        }
                      </strong>
                    </article>


                    <article
                      className={
                        styles.metricCard
                      }
                    >
                      <span>
                        Analyses exécutées
                      </span>

                      <strong>
                        {
                          report.inventory
                            .executed_analysis_count
                        }
                      </strong>
                    </article>


                    <article
                      className={
                        styles.metricCard
                      }
                    >
                      <span>
                        Contrôles qualité
                      </span>

                      <strong>
                        {
                          report.inventory
                            .quality_check_count
                        }
                      </strong>
                    </article>
                  </div>





                        )
                      : null
                  }


                  {
                    activeStep ===
                      "report" &&
                    signalKpis.length >
                    0
                      ? (
                          <>
                            <div
                              className={
                                styles.sectionHead
                              }
                            >
                              <h2>
                                Signaux clés
                              </h2>
                            </div>


                            <div
                              className={
                                styles.metricGrid
                              }
                            >
                              {
                                signalKpis.map(
                                  (
                                    kpi
                                  ) => (
                                    <article
                                      className={
                                        styles.metricCard
                                      }
                                      key={
                                        kpi.key
                                      }
                                    >
                                      <span>
                                        {
                                          kpi.label
                                        }
                                      </span>

                                      <strong>
                                        {
                                          kpi.value
                                        }
                                      </strong>

                                      <small>
                                        {
                                          kpi.context
                                        }
                                      </small>
                                    </article>
                                  )
                                )
                              }
                            </div>
                          </>
                        )
                      : null
                  }


                  {
                    activeStep ===
                      "report"
                      ? (
                          <QualityReportSection
                            report={
                              qualityReport
                            }
                            cleaningPlan={
                              cleaningPlan
                            }
                            cleaningExecution={
                              cleaningExecution
                            }
                          />
                        )
                      : null
                  }


                  {
                    activeStep ===
                      "report" &&
                    report.executive_summary
                      .length >
                    0
                      ? (
                          <section
                            className={
                              styles.summaryPanel
                            }
                          >
                            <div
                              className={
                                styles.summaryItem
                              }
                            >
                              <span>
                                Synthèse
                              </span>

                              {
                                report.executive_summary.map(
                                  (
                                    item
                                  ) => (
                                    <p
                                      key={
                                        item
                                      }
                                    >
                                      {
                                        item
                                      }
                                    </p>
                                  )
                                )
                              }
                            </div>
                          </section>
                        )
                      : null
                  }

                  {
                    activeStep ===
                      "report"
                      ? (
                          <ReportSelectionPanel
                            ragContextByAnalysisId={ragContextByAnalysisId}
                            reportAvailableAnalysisById={reportAvailableAnalysisById}
                            reportSelectionLoading={reportSelectionLoading}
                            requestedResolutionErrors={requestedResolutionErrors}
                            requestedResolutionLoadingId={requestedResolutionLoadingId}
                            selectedPromptAnalyses={selectedPromptAnalyses}
                            unresolvedDocumentRequests={unresolvedDocumentRequests}
                            unselectedAutomaticAnalyses={unselectedAutomaticAnalyses}
                            unselectedRequestedAnalyses={unselectedRequestedAnalyses}
                            handleReconfigureRequestedTimeSeries={handleReconfigureRequestedTimeSeries}
                            handleResolveRequestedRanking={handleResolveRequestedRanking}
                            handleResolveRequestedTimeSeries={handleResolveRequestedTimeSeries}
                            removePromptAnalysisFromReport={removePromptAnalysisFromReport}
                            setAvailableAnalysisReportSelection={setAvailableAnalysisReportSelection}
                          />
                        )
                      : null
                  }


                  {
                    activeStep ===
                      "analyses"
                      ? (
                          <>
                            {
                              report.entity_outlier_finding
                                ? (
                                    <EntityOutlierRequestedAnswer
                                      finding={
                                        report.entity_outlier_finding
                                      }
                                      objective={
                                        objective
                                      }
                                    />
                                  )
                                : aiNativeReport
                                  ? (
                                      <SelectableNativeAnalysisResult
                                        report={
                                          aiNativeReport
                                        }
                                        objective={
                                          aiNativeReport
                                              .planner
                                              .objective ||
                                            objective
                                        }
                                        includedInReport={
                                          initialPromptIncludedInReport
                                        }
                                        reportSelectionLoading={
                                          reportSelectionLoading
                                        }
                                        selectionCopy={{
                                          sourceLabel: "Demande initiale",
                                          includedMessage: "Cette analyse sera reprise dans le rapport.",
                                          excludedMessage: "Cette analyse restera dans l’espace d’exploration.",
                                          addLabel: "Ajouter au rapport",
                                          removeLabel: "Retirer du rapport",
                                        }}
                                        onToggleReportSelection={
                                          () => {
                                                        if (
                                                          !aiNativeReport
                                                        ) {
                                                          return;
                                                        }


                                                        void setPromptAnalysisReportSelection(
                                                          {
                                                            report:
                                                              aiNativeReport,

                                                            included:
                                                              !initialPromptIncludedInReport,
                                                          }
                                                        );
                                                      }
                                        }
                                      />
                                    )
                                  : aiPlanReport
                                    ? (
                                        <PlannerBlockedAnalysisCard
                                          planner={
                                            aiPlanReport
                                          }
                                          objective={
                                            objective
                                          }
                                        />
                                      )
                                    : null
                            }


                            <section
                              className={
                                styles.analysisFollowUpPanel
                              }
                              aria-labelledby="analysis-follow-up-title"
                            >
                              <div
                                className={
                                  styles.analysisFollowUpHeader
                                }
                              >
                                <div>
                                  <span
                                    className={
                                      styles.eyebrow
                                    }
                                  >
                                    IA locale · nouvelle demande
                                  </span>

                                  <h2
                                    id="analysis-follow-up-title"
                                    className={
                                      styles.analysisFollowUpTitle
                                    }
                                  >
                                    Poursuivre l’analyse
                                  </h2>

                                  <p
                                    className={
                                      styles.analysisFollowUpDescription
                                    }
                                  >
                                    Posez une autre question sur les mêmes données
                                    préparées. Chaque demande repasse par le planner,
                                    la validation Python et les outils déterministes.
                                  </p>
                                </div>

                                <span
                                  className={
                                    styles.analysisFollowUpStatus
                                  }
                                >
                                  {
                                    analysisFollowUpTurns.length
                                  }
                                  {" question"}
                                  {
                                    analysisFollowUpTurns.length >
                                    1
                                      ? "s"
                                      : ""
                                  }
                                  {" de suivi"}
                                </span>
                              </div>


                              <form
                                className={
                                  styles.analysisFollowUpForm
                                }
                                onSubmit={
                                  handleAnalysisFollowUpSubmit
                                }
                              >
                                <label
                                  htmlFor="analysis-follow-up-prompt"
                                  className={
                                    styles.analysisFollowUpLabel
                                  }
                                >
                                  Que souhaitez-vous analyser ensuite ?
                                </label>

                                <textarea
                                  id="analysis-follow-up-prompt"
                                  className={
                                    styles.analysisFollowUpInput
                                  }
                                  value={
                                    analysisFollowUpPrompt
                                  }
                                  onChange={
                                    (
                                      event
                                    ) =>
                                      setAnalysisFollowUpPrompt(
                                        event.target.value
                                      )
                                  }
                                  placeholder="Ex. Quelle catégorie a le prix unitaire moyen le plus élevé ?"
                                  rows={
                                    3
                                  }
                                  disabled={
                                    analysisFollowUpLoading
                                  }
                                />

                                <div
                                  className={
                                    styles.analysisFollowUpActions
                                  }
                                >
                                  <p
                                    className={
                                      styles.analysisFollowUpHint
                                    }
                                  >
                                    v0.1 · chaque prompt est interprété indépendamment.
                                    Pour une question de suivi, nommez encore la métrique
                                    ou la variable concernée.
                                  </p>

                                  <button
                                    type="submit"
                                    className={
                                      styles.analysisFollowUpButton
                                    }
                                    disabled={
                                      analysisFollowUpLoading ||
                                      aiNativeLoading ||
                                      !analysisFollowUpPrompt
                                        .trim() ||
                                      !preparationReadyForAnalysis
                                    }
                                  >
                                    {
                                      analysisFollowUpLoading
                                        ? "Analyse en cours…"
                                        : "Analyser"
                                    }
                                  </button>
                                </div>
                              </form>


                              {
                                analysisFollowUpError
                                  ? (
                                      <div
                                        className={
                                          styles.analysisFollowUpError
                                        }
                                        role="alert"
                                      >
                                        <strong>
                                          Nouvelle analyse impossible
                                        </strong>

                                        <span>
                                          {
                                            analysisFollowUpError
                                          }
                                        </span>
                                      </div>
                                    )
                                  : null
                              }


                              {
                                analysisFollowUpTurns.length >
                                1
                                  ? (
                                      <AnalysisFollowUpHistory
                                        turns={
                                          analysisFollowUpTurns
                                        }
                                        reportSelectionLoading={
                                          reportSelectionLoading
                                        }
                                        onToggleReportSelection={
                                          toggleFollowUpReportSelection
                                        }
                                      />
                                    )
                                  : null
                              }


                              {
                                latestAnalysisFollowUp
                                  ? (
                                      <SelectableNativeAnalysisResult
                                        report={
                                          latestAnalysisFollowUp.report
                                        }
                                        objective={
                                          latestAnalysisFollowUp.objective
                                        }
                                        includedInReport={
                                          latestAnalysisFollowUp
                                                        .included_in_report
                                        }
                                        reportSelectionLoading={
                                          reportSelectionLoading
                                        }
                                        selectionCopy={{
                                          sourceLabel: "Question de suivi",
                                          includedMessage: "Cette analyse sera reprise dans le rapport.",
                                          excludedMessage: "Cette analyse n’est pas ajoutée au rapport.",
                                          addLabel: "Ajouter au rapport",
                                          removeLabel: "Retirer du rapport",
                                        }}
                                        onToggleReportSelection={
                                          () =>
                                                        toggleFollowUpReportSelection(
                                                          latestAnalysisFollowUp.id
                                                        )
                                        }
                                        className={
                                          styles.analysisFollowUpResult
                                        }
                                        ariaLive="polite"
                                      />
                                    )
                                  : null
                              }
                            </section>


                            <SelectedPromptAnalysesSection
  selectedPromptAnalyses={selectedPromptAnalyses}
  removePromptAnalysisFromReport={removePromptAnalysisFromReport}
  reportSelectionError={reportSelectionError}
  reportSelectionLoading={reportSelectionLoading}
/>


                            {
                              (
                                documentSummary !==
                                  null ||
                                requestedPlan !==
                                  null ||
                                report.requested_findings.length >
                                  0
                              )
                                ? (
                                    <RequestedFindingsSection
                                      documentSummary={documentSummary}
                                      requestedPlan={requestedPlan}
                                      requestedFindings={report.requested_findings}
                                      reportAvailableAnalysisById={reportAvailableAnalysisById}
                                      ragContextByAnalysisId={ragContextByAnalysisId}
                                      requestedResolutionErrors={requestedResolutionErrors}
                                      requestedResolutionLoadingId={requestedResolutionLoadingId}
                                      handleReconfigureRequestedTimeSeries={handleReconfigureRequestedTimeSeries}
                                    />
                                  )
                                : null
                            }


                            {
                              report.main_findings.length >
                              0
                                ? (
                                    <MainFindingsSection
                                      findings={
                                        report.main_findings
                                      }
                                      ragContextByAnalysisId={
                                        ragContextByAnalysisId
                                      }
                                    />
                                  )
                                : null
                            }


                            <AnalysisAuditDisclosure
  report={report}
  ragReport={ragReport}
/>
                          </>
                        )
                      : null
                  }


                  {
                    activeStep ===
                      "report"
                      ? (
                          <ReportSynthesisActions
  reportSelectionDetails={reportSelectionDetails}
  pdfExportLoading={pdfExportLoading}
  handlePdfExport={handlePdfExport}
  setActiveStep={setActiveStep}
/>
                        )
                      : (
                          <div
                            className={
                              styles.submitArea
                            }
                          >
                            <div
                              className={
                                styles.submitInfo
                              }
                            >
                              <strong>
                                Analyse complète
                              </strong>

                              <span>
                                Passez au rapport pour une lecture
                                plus courte orientée décision.
                              </span>
                            </div>

                            <button
                              className={
                                styles.submitButton
                              }
                              type="button"
                              onClick={
                                () =>
                                  setActiveStep(
                                    "report"
                                  )
                              }
                            >
                              Voir le rapport
                            </button>
                          </div>
                        )
                  }
                </section>
              )
            : null
        }


        <footer
          className={
            styles.footer
          }
        >
          <strong>
            DataLens
          </strong>

          <span>
            Python déterministe
            {" · "}
            IA locale
            {" · "}
            preuves vérifiables
          </span>
        </footer>
      </div>
    </main>
  );
}
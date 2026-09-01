import {
  useState,
} from "react";

import type {
  PreparationUiStateView,
  SemanticCleaningChoiceView,
  SemanticCleaningExecutionView,
  SemanticCleaningPlanView,
  SemanticConfirmationReportView,
  SemanticReviewReportView,
  PreparationSessionView,
  SemanticCleaningApplyResponseView,
} from "./preparationTypes";


type SemanticPreparationStateDependencies = {
  apiUrl:
    string;

  clearFinalValidationError:
    () => void;

  resetFinalValidationUiState:
    () => void;
};


type BuildSemanticCleaningPlanInput = {
  review:
    SemanticReviewReportView;

  preparationSession:
    PreparationSessionView |
    null;

  datasetFiles:
    File[];

  appliedCleaningActionIds:
    string[];
};


// DATALENS_SEMANTIC_PARTIAL_DECISION_HOOK_V0_1
export function useSemanticPreparationState({
  apiUrl,
  clearFinalValidationError,
  resetFinalValidationUiState,
}: SemanticPreparationStateDependencies) {
  const [
    semanticReview,
    setSemanticReview,
  ] =
    useState<
      SemanticReviewReportView |
      null
    >(
      null
    );

  const [
    semanticReviewLoading,
    setSemanticReviewLoading,
  ] =
    useState(
      false
    );

  const [
    semanticReviewError,
    setSemanticReviewError,
  ] =
    useState<
      string |
      null
    >(
      null
    );

  const [
    semanticCleaningPlan,
    setSemanticCleaningPlan,
  ] =
    useState<
      SemanticCleaningPlanView |
      null
    >(
      null
    );

  const [
    semanticPlanLoading,
    setSemanticPlanLoading,
  ] =
    useState(
      false
    );

  const [
    semanticPlanError,
    setSemanticPlanError,
  ] =
    useState<
      string |
      null
    >(
      null
    );

  const [
    selectedSemanticActionIds,
    setSelectedSemanticActionIds,
  ] =
    useState<
      string[]
    >(
      []
    );

  const [
    semanticCanonicalValues,
    setSemanticCanonicalValues,
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
    semanticApplyLoading,
    setSemanticApplyLoading,
  ] =
    useState(
      false
    );

  const [
    semanticApplyError,
    setSemanticApplyError,
  ] =
    useState<
      string |
      null
    >(
      null
    );

  const [
    semanticCleaningExecution,
    setSemanticCleaningExecution,
  ] =
    useState<
      SemanticCleaningExecutionView |
      null
    >(
      null
    );

  const [
    appliedSemanticChoices,
    setAppliedSemanticChoices,
  ] =
    useState<
      SemanticCleaningChoiceView[]
    >(
      []
    );

  const [
    confirmedSemanticIssueIds,
    setConfirmedSemanticIssueIds,
  ] =
    useState<
      string[]
    >(
      []
    );

  const [
    semanticManualResolutionNotes,
    setSemanticManualResolutionNotes,
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
    semanticConfirmation,
    setSemanticConfirmation,
  ] =
    useState<
      SemanticConfirmationReportView |
      null
    >(
      null
    );

  const [
    semanticConfirmationLoading,
    setSemanticConfirmationLoading,
  ] =
    useState(
      false
    );

  const [
    semanticConfirmationError,
    setSemanticConfirmationError,
  ] =
    useState<
      string |
      null
    >(
      null
    );


  function restoreFromPreparationUiState(
    restoredUiState:
      PreparationUiStateView
  ) {
    setSemanticReview(
      restoredUiState
        .semantic_review
    );

    setSemanticCleaningPlan(
      restoredUiState
        .semantic_cleaning_plan
    );

    setSemanticCleaningExecution(
      restoredUiState
        .semantic_cleaning_execution
    );

    const restoredSemanticChoices =
      restoredUiState
        .applied_semantic_choices;

    setAppliedSemanticChoices(
      restoredSemanticChoices
    );

    setSelectedSemanticActionIds(
      restoredUiState
        .semantic_cleaning_execution
        ? restoredSemanticChoices.map(
            (
              choice
            ) =>
              choice.action_id
          )
        : []
    );

    const restoredCanonicalValues:
      Record<
        string,
        string
      > =
        {};

    for (
      const action
      of (
        restoredUiState
          .semantic_cleaning_plan
          ?.actions ??
        []
      )
    ) {
      restoredCanonicalValues[
        action.action_id
      ] =
        action
          .suggested_canonical_value;
    }

    for (
      const choice
      of restoredSemanticChoices
    ) {
      restoredCanonicalValues[
        choice.action_id
      ] =
        choice.canonical_value;
    }

    setSemanticCanonicalValues(
      restoredCanonicalValues
    );

    setConfirmedSemanticIssueIds(
      restoredUiState
        .confirmed_semantic_issue_ids
    );

    const restoredManualResolutionNotes:
      Record<
        string,
        string
      > =
        {};

    for (
      const resolution
      of restoredUiState
        .semantic_manual_resolutions
    ) {
      if (
        resolution.issue_id &&
        resolution.note
      ) {
        restoredManualResolutionNotes[
          resolution.issue_id
        ] =
          resolution.note;
      }
    }

    setSemanticManualResolutionNotes(
      restoredManualResolutionNotes
    );

    setSemanticConfirmation(
      restoredUiState
        .semantic_confirmation
    );
  }


  function handleSetSemanticDecision(
    actionId:
      string,

    shouldMerge:
      boolean
  ) {
setSelectedSemanticActionIds(
      (
        current
      ) => {
        if (
          shouldMerge
        ) {
          return current.includes(
            actionId
          )
            ? current
            : [
                ...current,
                actionId,
              ];
        }


        return current.filter(
          (
            value
          ) =>
            value !==
            actionId
        );
      }
    );

    setSemanticApplyError(
      null
    );

    setSemanticConfirmation(
      null
    );

    setSemanticConfirmationError(
      null
    );

    clearFinalValidationError();
  }


  function handleSemanticCanonicalChange(
    actionId:
      string,

    canonicalValue:
      string
  ) {
setSemanticCanonicalValues(
      (
        current
      ) => ({
        ...current,

        [actionId]:
          canonicalValue,
      })
    );

    setSemanticApplyError(
      null
    );

    setSemanticConfirmation(
      null
    );

    setSemanticConfirmationError(
      null
    );

    clearFinalValidationError();
  }


  function handleToggleSemanticIssueConfirmation(
    issueId:
      string,

    checked:
      boolean
  ) {
    setConfirmedSemanticIssueIds(
      (
        current
      ) => {
        if (
          checked
        ) {
          return current.includes(
            issueId
          )
            ? current
            : [
                ...current,
                issueId,
              ];
        }


        return current.filter(
          (
            value
          ) =>
            value !==
            issueId
        );
      }
    );


    setSemanticConfirmation(
      null
    );

    setSemanticConfirmationError(
      null
    );

    clearFinalValidationError();
  }


  function handleSemanticManualResolutionChange(
    issueId:
      string,

    note:
      string
  ) {
    setSemanticManualResolutionNotes(
      (
        current
      ) => ({
        ...current,

        [issueId]:
          note,
      })
    );


    setSemanticConfirmation(
      null
    );

    setSemanticConfirmationError(
      null
    );

    clearFinalValidationError();
  }


  function resetSemanticPreparation() {
    setSemanticReview(
      null
    );

    setSemanticReviewError(
      null
    );

    setSemanticCleaningPlan(
      null
    );

    setSemanticPlanError(
      null
    );

    setSelectedSemanticActionIds(
      []
    );

    setSemanticCanonicalValues(
      {}
    );

    setSemanticCleaningExecution(
      null
    );

    setAppliedSemanticChoices(
      []
    );

    setSemanticApplyError(
      null
    );

    setConfirmedSemanticIssueIds(
      []
    );

    setSemanticManualResolutionNotes(
      {}
    );

    setSemanticConfirmation(
      null
    );

    setSemanticConfirmationLoading(
      false
    );

    setSemanticConfirmationError(
      null
    );

    resetFinalValidationUiState();
  }


    async function buildSemanticCleaningPlan({
    review,
    preparationSession,
    datasetFiles,
    appliedCleaningActionIds,
  }: BuildSemanticCleaningPlanInput) {
    if (
      preparationSession ===
      null
    ) {
      setSemanticPlanError(
        "Aucune session de préparation active."
      );

      return;
    }


    setSemanticPlanLoading(
      true
    );

    setSemanticPlanError(
      null
    );

    setSemanticCleaningPlan(
      null
    );

    setSelectedSemanticActionIds(
      []
    );

    setSemanticCleaningExecution(
      null
    );

    setAppliedSemanticChoices(
      []
    );


    try {
      const formData =
        new FormData();


      for (
        const file
        of datasetFiles
      ) {
        formData.append(
          "dataset_files",
          file
        );
      }


      formData.append(
        "workflow_id",
        preparationSession.workflow_id
      );


      if (
        appliedCleaningActionIds.length >
        0
      ) {
        formData.append(
          "approved_action_ids_json",
          JSON.stringify(
            appliedCleaningActionIds
          )
        );
      }


      formData.append(
        "semantic_decisions_json",
        JSON.stringify(
          review.decisions
        )
      );


      const response =
        await fetch(
          `${apiUrl}/preparation/semantic-cleaning-plan`,
          {
            method:
              "POST",

            body:
              formData,
          }
        );


      const payload =
        await response.json();


      if (
        !response.ok
      ) {
        const detail =
          typeof payload.detail ===
          "string"
            ? payload.detail
            : JSON.stringify(
                payload.detail ??
                payload
              );


        throw new Error(
          detail
        );
      }


      const typedPlan =
        payload as
          SemanticCleaningPlanView;


      setSemanticCleaningPlan(
        typedPlan
      );


      const canonicalValues: Record<
        string,
        string
      > = {};


      for (
        const action
        of typedPlan.actions
      ) {
        canonicalValues[
          action.action_id
        ] =
          action
            .suggested_canonical_value;
      }


      setSemanticCanonicalValues(
        canonicalValues
      );
    } catch (
      caughtError
    ) {
      setSemanticCleaningPlan(
        null
      );

      setSemanticPlanError(
        caughtError
          instanceof Error
          ? caughtError.message
          : "Impossible de reconstruire le plan sémantique."
      );
    } finally {
      setSemanticPlanLoading(
        false
      );
    }
  }


    function beginSemanticReviewRun() {
  setSemanticReviewLoading(
        true
      );

      setSemanticReviewError(
        null
      );

      setSemanticReview(
        null
      );

      setSemanticCleaningPlan(
        null
      );

      setSemanticPlanError(
        null
      );

      setSelectedSemanticActionIds(
        []
      );

      setSemanticCanonicalValues(
        {}
      );

      setSemanticCleaningExecution(
        null
      );

      setAppliedSemanticChoices(
        []
      );

      setConfirmedSemanticIssueIds(
        []
      );

      setSemanticManualResolutionNotes(
        {}
      );

      setSemanticConfirmation(
        null
      );

      setSemanticConfirmationError(
        null
      );

      clearFinalValidationError();
  }


    function prepareSemanticConfirmationRun(
    review:
      SemanticReviewReportView
  ) {
  const manualResolutions =
        review.decisions
          .filter(
            (
              decision
            ) =>
              (
                decision.verdict ===
                  "abstain" ||
                decision.verdict ===
                  "flag_for_review"
              ) &&
              confirmedSemanticIssueIds.includes(
                decision.issue_id
              )
          )
          .map(
            (
              decision
            ) => ({
              issue_id:
                decision.issue_id,

              note:
                semanticManualResolutionNotes[
                  decision.issue_id
                ]?.trim() ??
                "",
            })
          )
          .filter(
            (
              resolution
            ) =>
              resolution.note.length >=
              3
          );


      setSemanticConfirmationLoading(
        true
      );

      setSemanticConfirmationError(
        null
      );

      clearFinalValidationError();

    return (
      manualResolutions
    );
  }


    function prepareSemanticCleaningApply({
    workflowId,
  }: {
    workflowId:
      string |
      null;
  }): {
    workflowId: string;
    semanticDecisions:
      SemanticReviewReportView["decisions"];
    choices:
      SemanticCleaningChoiceView[];
  } |
    null {
    if (
      semanticReview ===
      null
      ||
      semanticCleaningPlan ===
      null
    ) {
      setSemanticApplyError(
        "Aucun plan sémantique n’est disponible."
      );

      return null;
    }


    if (
      workflowId ===
      null
    ) {
      setSemanticApplyError(
        "Aucune session de préparation active."
      );

      return null;
    }


    if (
      selectedSemanticActionIds.length ===
      0
    ) {
      setSemanticApplyError(
        "Sélectionnez au moins une fusion à appliquer."
      );

      return null;
    }

    let choices:
      SemanticCleaningChoiceView[];


    try {
      choices =
        selectedSemanticActionIds.map(
          (
            actionId
          ) => {
            const canonicalValue =
              semanticCanonicalValues[
                actionId
              ];


            if (
              !canonicalValue
            ) {
              throw new Error(
                "Une valeur canonique manque pour une fusion sélectionnée."
              );
            }


            return {
              action_id:
                actionId,

              canonical_value:
                canonicalValue,
            };
          }
        );
    }
    catch (
      caughtError
    ) {
      setSemanticApplyError(
        caughtError
          instanceof Error
          ? caughtError.message
          : "Une valeur canonique est invalide."
      );

      return null;
    }


    setSemanticApplyLoading(
      true
    );

    setSemanticApplyError(
      null
    );


        return {
      workflowId,

      semanticDecisions:
        semanticReview.decisions,

      choices,
    };
  }


    function completeSemanticCleaningApply({
    response,
    choices,
  }: {
    response:
      SemanticCleaningApplyResponseView;
    choices:
      SemanticCleaningChoiceView[];
  }) {
    setSemanticCleaningPlan(
      response.plan
    );

    setSemanticCleaningExecution(
      response.execution
    );

    setAppliedSemanticChoices(
      choices
    );

    setSemanticConfirmation(
      null
    );

    setSemanticConfirmationError(
      null
    );
  }


  function failSemanticCleaningApply(
    caughtError: unknown
  ) {
    setSemanticCleaningExecution(
      null
    );

    setAppliedSemanticChoices(
      []
    );

    setSemanticApplyError(
      caughtError instanceof Error
        ? caughtError.message
        : "Le nettoyage sémantique a échoué."
    );
  }


  function finishSemanticCleaningApply() {
    setSemanticApplyLoading(
      false
    );
  }


    return {
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
  };
}

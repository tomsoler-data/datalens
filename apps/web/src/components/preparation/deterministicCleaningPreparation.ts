type CleaningPlanInput = {
  actions: Array<{
    action_id: string;
  }>;
};


type PreparationSessionInput = {
  workflow_id: string;
};


export type DeterministicCleaningPreparation =
  | {
      ok: false;
      error: string;
    }
  | {
      ok: true;
      workflowId: string;
      approvedCleaningActionIds: string[];
      rejectedCleaningActionIds: string[];
    };


export function prepareDeterministicCleaningApply({
  datasetFiles,
  cleaningPlan,
  preparationSession,
  selectedCleaningActionIds,
}: {
  datasetFiles: File[];
  cleaningPlan: CleaningPlanInput | null;
  preparationSession: PreparationSessionInput | null;
  selectedCleaningActionIds: string[];
}): DeterministicCleaningPreparation {
  if (
    datasetFiles.length === 0
  ) {
    return {
      ok: false,
      error: "Ajoutez au moins un fichier CSV.",
    };
  }


  if (
    cleaningPlan === null
  ) {
    return {
      ok: false,
      error: "Aucun plan de nettoyage n’est disponible.",
    };
  }


  if (
    preparationSession === null
  ) {
    return {
      ok: false,
      error: "La session de préparation est indisponible.",
    };
  }


  const approvedCleaningActionIdSet =
    new Set(
      selectedCleaningActionIds
    );


  const rejectedCleaningActionIds =
    cleaningPlan.actions
      .map(
        action =>
          action.action_id
      )
      .filter(
        actionId =>
          !approvedCleaningActionIdSet.has(
            actionId
          )
      );


  return {
    ok: true,

    workflowId:
      preparationSession.workflow_id,

    approvedCleaningActionIds:
      [...selectedCleaningActionIds],

    rejectedCleaningActionIds,
  };
}

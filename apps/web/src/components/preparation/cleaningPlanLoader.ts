import {
  getPreparationSession,
} from "./preparationApi";

import type {
  CleaningPlanView,
} from "./preparationTypes";


type SynchronizedPreparationSession =
  Awaited<
    ReturnType<
      typeof getPreparationSession
    >
  >;


type CleaningPlanLoaderDependencies = {
  apiUrl:
    string;

  setPreparationSession:
    (
      session:
        SynchronizedPreparationSession
    ) => void;

  setCleaningPlan:
    (
      plan:
        CleaningPlanView |
        null
    ) => void;

  setCleaningPlanLoading:
    (
      loading:
        boolean
    ) => void;

  setCleaningPlanError:
    (
      error:
        string |
        null
    ) => void;

  setSelectedCleaningActionIds:
    (
      actionIds:
        string[]
    ) => void;

  setCleaningExecution:
    (
      execution:
        null
    ) => void;

  setAppliedCleaningActionIds:
    (
      actionIds:
        string[]
    ) => void;
};


export function createCleaningPlanLoader({
  apiUrl,
  setPreparationSession,
  setCleaningPlan,
  setCleaningPlanLoading,
  setCleaningPlanError,
  setSelectedCleaningActionIds,
  setCleaningExecution,
  setAppliedCleaningActionIds,
}: CleaningPlanLoaderDependencies) {
  async function loadCleaningPlan(
    files:
      File[],

    workflowId:
      string
  ) {
    setCleaningPlanLoading(
      true
    );

    setCleaningPlanError(
      null
    );

    setCleaningPlan(
      null
    );

    setSelectedCleaningActionIds(
      []
    );

    setCleaningExecution(
      null
    );

    setAppliedCleaningActionIds(
      []
    );


    try {
      const formData =
        new FormData();


      for (
        const file
        of files
      ) {
        formData.append(
          "dataset_files",
          file
        );
      }


      formData.append(
        "workflow_id",
        workflowId
      );


      const response =
        await fetch(
          `${apiUrl}/preparation/cleaning-plan`,
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
          CleaningPlanView;


      setCleaningPlan(
        typedPlan
      );


      setSelectedCleaningActionIds(
        typedPlan.actions
          .filter(
            (
              action
            ) =>
              action.safe_candidate
          )
          .map(
            (
              action
            ) =>
              action.action_id
          )
      );


      const synchronizedSession =
        await getPreparationSession(
          workflowId
        );


      setPreparationSession(
        synchronizedSession
      );
    } catch (
      caughtError
    ) {
      setCleaningPlan(
        null
      );

      setCleaningPlanError(
        caughtError
          instanceof Error
          ? caughtError.message
          : "Impossible de construire le plan de nettoyage."
      );
    } finally {
      setCleaningPlanLoading(
        false
      );
    }
  }


  return loadCleaningPlan;
}

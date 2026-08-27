import type {
  ReportAvailableAnalysisDetailView,
} from "./analysisTypes";


type AvailableAnalysisReportSelectionDependencies = {
  apiUrl:
    string;

  hasPreparationSession:
    boolean;

  preparationWorkflowId:
    string;

  setReportSelectionLoading:
    (
      loading:
        boolean
    ) => void;

  setReportSelectionError:
    (
      error:
        string |
        null
    ) => void;

  refreshReportSelection:
    (
      workflowId:
        string
    ) => Promise<
      unknown
    >;
};


export function createAvailableAnalysisReportSelection({
  apiUrl,
  hasPreparationSession,
  preparationWorkflowId,
  setReportSelectionLoading,
  setReportSelectionError,
  refreshReportSelection,
}: AvailableAnalysisReportSelectionDependencies) {
  async function setAvailableAnalysisReportSelection(
    {
      analysis,
      included,
    }:
      {
        analysis:
          ReportAvailableAnalysisDetailView;

        included:
          boolean;
      }
  ) {
    if (
      !hasPreparationSession
    ) {
      setReportSelectionError(
        "La session de préparation est indisponible."
      );

      return;
    }


    const workflowId =
      preparationWorkflowId;


    setReportSelectionLoading(
      true
    );

    setReportSelectionError(
      null
    );


    try {
      const response =
        await fetch(
          `${apiUrl}/report/selection/${
            included
              ? "add"
              : "remove"
          }`,
          {
            method:
              "POST",

            headers:
              {
                "Content-Type":
                  "application/json",
              },

            body:
              JSON.stringify(
                {
                  workflow_id:
                    workflowId,

                  analysis_id:
                    analysis
                      .analysis_id,
                }
              ),
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


      await refreshReportSelection(
        workflowId
      );
    } catch (
      caughtError
    ) {
      setReportSelectionError(
        caughtError
          instanceof Error
          ? caughtError.message
          : "La sélection du rapport n’a pas pu être mise à jour."
      );
    } finally {
      setReportSelectionLoading(
        false
      );
    }
  }


  return setAvailableAnalysisReportSelection;
}

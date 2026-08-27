import type {
  AINativePipelineReportView,
} from "./analysisTypes";


type RefreshedReportSelectionView = {
  analyses:
    Array<{
      selection: {
        analysis_id:
          string;
      };
    }>;
};


type PromptAnalysisReportSelectionDependencies = {
  apiUrl:
    string;

  hasPreparationSession:
    boolean;

  preparationWorkflowId:
    string;

  aiNativeAnalysisId:
    string |
    null;

  setInitialPromptIncludedInReport:
    (
      included:
        boolean
    ) => void;

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
      RefreshedReportSelectionView |
      null
    >;
};


export function createPromptAnalysisReportSelection({
  apiUrl,
  hasPreparationSession,
  preparationWorkflowId,
  aiNativeAnalysisId,
  setInitialPromptIncludedInReport,
  setReportSelectionLoading,
  setReportSelectionError,
  refreshReportSelection,
}: PromptAnalysisReportSelectionDependencies) {
  async function setPromptAnalysisReportSelection(
    {
      report,
      included,
    }:
      {
        report:
          AINativePipelineReportView;

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


    const analysisId =
      report.analysis_id
        ?.trim();


    if (
      !analysisId
    ) {
      setReportSelectionError(
        "Cette analyse ne possède pas encore d’identifiant serveur stable."
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
                    analysisId,
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


      const refreshed =
        await refreshReportSelection(
          workflowId
        );


      const selected =
        Boolean(
          refreshed
            ?.analyses
            .some(
              (
                detail
              ) =>
                detail
                  .selection
                  .analysis_id
                ===
                analysisId
            )
        );


      if (
        aiNativeAnalysisId
        ===
        analysisId
      ) {
        setInitialPromptIncludedInReport(
          selected
        );
      }
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


  return setPromptAnalysisReportSelection;
}

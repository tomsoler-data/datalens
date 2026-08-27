type ReportPdfExportDependencies = {
  apiUrl:
    string;

  hasPreparationSession:
    boolean;

  workflowId:
    string |
    null;

  selectedCount:
    number;

  setPdfExportLoading:
    (
      loading:
        boolean
    ) => void;

  setError:
    (
      error:
        string |
        null
    ) => void;
};


export function createReportPdfExportHandler({
  apiUrl,
  hasPreparationSession,
  workflowId,
  selectedCount,
  setPdfExportLoading,
  setError,
}: ReportPdfExportDependencies) {
  async function handlePdfExport() {
    if (
      !hasPreparationSession
    ) {
      setError(
        "La session de préparation est indisponible."
      );

      return;
    }


    if (
      (
        selectedCount ??
        0
      ) ===
      0
    ) {
      setError(
        "Ajoutez au moins une analyse exécutée au rapport avant l’export PDF."
      );

      return;
    }


    setPdfExportLoading(
      true
    );

    setError(
      null
    );


    try {
      const response =
        await fetch(
          `${apiUrl}/report/export-pdf`,
          {
            method:
              "POST",

            headers: {
              "Content-Type":
                "application/json",
            },

            body:
              JSON.stringify(
                {
                  workflow_id:
                    workflowId,
                }
              ),
          }
        );


      if (
        !response.ok
      ) {
        const contentType =
          response.headers.get(
            "content-type"
          ) ??
          "";


        if (
          contentType.includes(
            "application/json"
          )
        ) {
          const payload =
            await response.json();

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


        throw new Error(
          await response.text() ||
          "La génération du PDF a échoué."
        );
      }


      const blob =
        await response.blob();


      const objectUrl =
        URL.createObjectURL(
          blob
        );


      const disposition =
        response.headers.get(
          "content-disposition"
        ) ??
        "";


      const filenameMatch =
        disposition.match(
          /filename="([^"]+)"/i
        );


      const date =
        new Date()
          .toISOString()
          .slice(
            0,
            10
          );


      const anchor =
        document.createElement(
          "a"
        );


      anchor.href =
        objectUrl;

      anchor.download =
        filenameMatch?.[1] ??
        `datalens-rapport-${date}.pdf`;

      document.body.appendChild(
        anchor
      );

      anchor.click();

      anchor.remove();


      URL.revokeObjectURL(
        objectUrl
      );
    } catch (
      caughtError
    ) {
      setError(
        caughtError
          instanceof Error
          ? caughtError.message
          : "Impossible de générer le PDF."
      );
    } finally {
      setPdfExportLoading(
        false
      );
    }
  }


  return handlePdfExport;
}

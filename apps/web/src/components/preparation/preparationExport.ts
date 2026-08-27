type PreparedDataExportDependencies = {
  apiUrl:
    string;

  datasetFiles:
    File[];

  appliedCleaningActionIds:
    string[];

  setPreparedExportLoading:
    (
      loading:
        boolean
    ) => void;

  setPreparedExportError:
    (
      error:
        string |
        null
    ) => void;
};


export function createPreparedDataExportHandler({
  apiUrl,
  datasetFiles,
  appliedCleaningActionIds,
  setPreparedExportLoading,
  setPreparedExportError,
}: PreparedDataExportDependencies) {
  async function handleExportPreparedData() {
    if (
      datasetFiles.length ===
      0
    ) {
      setPreparedExportError(
        "Ajoutez au moins un fichier CSV."
      );

      return;
    }


    if (
      appliedCleaningActionIds.length ===
      0
    ) {
      setPreparedExportError(
        "Appliquez d’abord les corrections sélectionnées."
      );

      return;
    }


    setPreparedExportLoading(
      true
    );

    setPreparedExportError(
      null
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
        "approved_action_ids_json",
        JSON.stringify(
          appliedCleaningActionIds
        )
      );


      const response =
        await fetch(
          `${apiUrl}/preparation/cleaning-export`,
          {
            method:
              "POST",

            body:
              formData,
          }
        );


      if (
        !response.ok
      ) {
        let detail =
          "L’export des données préparées a échoué.";


        try {
          const payload =
            await response.json();


          detail =
            typeof payload.detail ===
            "string"
              ? payload.detail
              : JSON.stringify(
                  payload.detail ??
                  payload
                );
        } catch {
          // Keep the generic message when the server
          // did not return JSON.
        }


        throw new Error(
          detail
        );
      }


      const blob =
        await response.blob();


      const disposition =
        response.headers.get(
          "content-disposition"
        ) ??
        "";


      const filenameMatch =
        disposition.match(
          /filename="?([^"]+)"?/i
        );


      const fallbackFilename =
        datasetFiles.length ===
        1
          ? `${datasetFiles[0].name.replace(
              /\.csv$/i,
              ""
            )}_prepared.csv`
          : "datalens_prepared_datasets.zip";


      const filename =
        filenameMatch?.[1] ??
        fallbackFilename;


      const objectUrl =
        URL.createObjectURL(
          blob
        );


      const link =
        document.createElement(
          "a"
        );


      link.href =
        objectUrl;

      link.download =
        filename;

      document.body.appendChild(
        link
      );

      link.click();

      link.remove();


      URL.revokeObjectURL(
        objectUrl
      );
    } catch (
      caughtError
    ) {
      setPreparedExportError(
        caughtError
          instanceof Error
          ? caughtError.message
          : "L’export des données préparées a échoué."
      );
    } finally {
      setPreparedExportLoading(
        false
      );
    }
  }


  return handleExportPreparedData;
}

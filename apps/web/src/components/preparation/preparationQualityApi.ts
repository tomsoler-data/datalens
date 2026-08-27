import type {
  DataQualityReportView,
} from "./preparationTypes";


const API_URL =
  process.env.NEXT_PUBLIC_DATALENS_API_URL ??
  "http://127.0.0.1:8000";


export async function runPreparationQuality({
  files,
  workflowId,
}: {
  files: File[];
  workflowId: string;
}): Promise<DataQualityReportView> {
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
      `${API_URL}/preparation/quality`,
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


  return (
    payload as
      DataQualityReportView
  );
}

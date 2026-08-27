const API_URL =
  process.env.NEXT_PUBLIC_DATALENS_API_URL ??
  "http://127.0.0.1:8000";


export type AnalysisRunTransportResult = {
  contextualized:
    boolean;

  payload:
    unknown;
};


export async function runAnalysisRequest({
  workflowId,
  objective,
  documents,
}: {
  workflowId: string;
  objective: string;
  documents: readonly File[];
}): Promise<AnalysisRunTransportResult> {
  const formData =
    new FormData();


  formData.append(
    "workflow_id",
    workflowId
  );


  if (
    objective.trim()
  ) {
    formData.append(
      "objective",
      objective.trim()
    );
  }


  /*
  * Preparation has already crossed VALIDATE here.
  *
  * Analysis must not receive cleaning or semantic-cleaning
  * overrides. The backend Analysis Input Handoff loads the
  * exact server-owned artifacts selected and certified during
  * Preparation.
  */

  const contextualized =
    documents.length > 0;


  if (
    contextualized
  ) {
    for (
      const document
      of documents
    ) {
      formData.append(
        "document_files",
        document
      );
    }


    formData.append(
      "rag_top_k",
      "3"
    );


    formData.append(
      "embedding_model",
      "embeddinggemma"
    );
  }


  const endpoint =
    contextualized
      ? "/analysis/run-contextualized"
      : "/analysis/run";


  const response =
    await fetch(
      `${API_URL}${endpoint}`,
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


  return {
    contextualized,
    payload,
  };
}

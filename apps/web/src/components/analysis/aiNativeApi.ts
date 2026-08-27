import type {
  AINativePipelineReportView,
} from "./analysisTypes";


const API_URL =
  process.env.NEXT_PUBLIC_DATALENS_API_URL ??
  "http://127.0.0.1:8000";


export async function runAiNativeAnalysis({
  workflowId,
  objective,
  plannerModel,
  toolModel,
}: {
  workflowId: string;
  objective: string;
  plannerModel: string;
  toolModel: string;
}): Promise<AINativePipelineReportView> {
  const formData =
    new FormData();


  formData.append(
    "workflow_id",
    workflowId
  );


  formData.append(
    "objective",
    objective
  );


  formData.append(
    "planner_model",
    plannerModel
  );


  formData.append(
    "tool_model",
    toolModel
  );


  const response =
    await fetch(
      `${API_URL}/planning/ai-native-run`,
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
      AINativePipelineReportView
  );
}

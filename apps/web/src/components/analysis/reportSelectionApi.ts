import type {
  ReportAvailableAnalysisListView,
  ReportSelectionDetailsView,
} from "./analysisTypes";


const API_URL =
  process.env.NEXT_PUBLIC_DATALENS_API_URL ??
  "http://127.0.0.1:8000";


export type ReportSelectionState = {
  selection:
    ReportSelectionDetailsView;

  available:
    ReportAvailableAnalysisListView;
};


export async function loadReportSelectionState(
  workflowId: string
): Promise<ReportSelectionState> {
  const [
    selectionResponse,
    availableResponse,
  ] =
    await Promise.all([
      fetch(
        `${API_URL}/report/selection/details?workflow_id=${encodeURIComponent(
          workflowId
        )}`,
        {
          method:
            "GET",

          cache:
            "no-store",
        }
      ),

      fetch(
        `${API_URL}/report/analyses/details?workflow_id=${encodeURIComponent(
          workflowId
        )}`,
        {
          method:
            "GET",

          cache:
            "no-store",
        }
      ),
    ]);


  const selectionPayload =
    await selectionResponse.json();


  const availablePayload =
    await availableResponse.json();


  if (
    !selectionResponse.ok
  ) {
    const detail =
      typeof selectionPayload.detail ===
        "string"
        ? selectionPayload.detail
        : JSON.stringify(
            selectionPayload.detail ??
            selectionPayload
          );


    throw new Error(
      detail
    );
  }


  if (
    !availableResponse.ok
  ) {
    const detail =
      typeof availablePayload.detail ===
        "string"
        ? availablePayload.detail
        : JSON.stringify(
            availablePayload.detail ??
            availablePayload
          );


    throw new Error(
      detail
    );
  }


  return {
    selection:
      selectionPayload as
        ReportSelectionDetailsView,

    available:
      availablePayload as
        ReportAvailableAnalysisListView,
  };
}

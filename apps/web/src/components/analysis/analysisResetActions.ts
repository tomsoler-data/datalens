type NullSetter =
  (value: null) => void;


export function createAnalysisOutputReset({
  setReport,
  setRagReport,
  setDocumentSummary,
  setRequestedPlan,
  setAiPlanReport,
  setAiPlanError,
  setAiNativeReport,
  setAiNativeError,
}: {
  setReport: NullSetter;
  setRagReport: NullSetter;
  setDocumentSummary: NullSetter;
  setRequestedPlan: NullSetter;
  setAiPlanReport: NullSetter;
  setAiPlanError: NullSetter;
  setAiNativeReport: NullSetter;
  setAiNativeError: NullSetter;
}): () => void {
  return () => {
    setReport(null);
    setRagReport(null);
    setDocumentSummary(null);
    setRequestedPlan(null);
    setAiPlanReport(null);
    setAiPlanError(null);
    setAiNativeReport(null);
    setAiNativeError(null);
  };
}

import type {
  AINativePipelineReportView,
} from "./analysisTypes";


export function nativePipelineHasExecutedResult(
  report:
    AINativePipelineReportView |
    null
): boolean {
  if (
    report ===
    null
  ) {
    return false;
  }


  return (
    report.executed_count >
      0 &&
    report.items.some(
      (
        item
      ) =>
        item.pipeline_status ===
          "executed" &&
        item.native_tool
          ?.execution
          ?.result
          !==
          null
    )
  );
}


export function toolEngineLabel(
  model:
    string |
    null |
    undefined
): string {
  if (
    !model
  ) {
    return "IA locale";
  }


  if (
    model
      .toLowerCase()
      .includes(
        "qwen"
      )
  ) {
    return "Qwen · tool calling local";
  }


  return model;
}

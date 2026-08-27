import type {
  AINativePipelineReportView,
} from "./analysisTypes";

import {
  nativePipelineHasExecutedResult,
} from "./analysisExecutionPresentation";

import NativeRequestedAnalysisCard
  from "./NativeRequestedAnalysisCard";


type SelectionCopy = {
  sourceLabel: string;
  includedMessage: string;
  excludedMessage: string;
  addLabel: string;
  removeLabel: string;
};


type SelectableNativeAnalysisResultProps = {
  report: AINativePipelineReportView;
  objective: string;
  includedInReport: boolean;
  reportSelectionLoading: boolean;
  selectionCopy: SelectionCopy;
  onToggleReportSelection: () => void | Promise<void>;
  className?: string;
  ariaLive?: "off" | "polite" | "assertive";
};


export default function SelectableNativeAnalysisResult({
  report,
  objective,
  includedInReport,
  reportSelectionLoading,
  selectionCopy,
  onToggleReportSelection,
  className,
  ariaLive,
}: SelectableNativeAnalysisResultProps) {
  return (
    <div
      className={
        className
      }
      aria-live={
        ariaLive
      }
    >
      {
        nativePipelineHasExecutedResult(
          report
        )
          ? (
              <div
                style={{
                  display:
                    "flex",

                  alignItems:
                    "center",

                  justifyContent:
                    "space-between",

                  gap:
                    "14px",

                  margin:
                    "0 0 10px",

                  padding:
                    "10px 12px",

                  border:
                    "1px solid rgba(154, 174, 204, 0.12)",

                  borderRadius:
                    "11px",

                  background:
                    "rgba(3, 8, 17, 0.30)",
                }}
              >
                <div>
                  <strong
                    style={{
                      display:
                        "block",

                      fontSize:
                        "0.76rem",

                      fontWeight:
                        650,
                    }}
                  >
                    {
                      selectionCopy
                        .sourceLabel
                    }
                  </strong>

                  <span
                    style={{
                      display:
                        "block",

                      marginTop:
                        "3px",

                      color:
                        "#9eacc0",

                      fontSize:
                        "0.68rem",
                    }}
                  >
                    {
                      includedInReport
                        ? selectionCopy
                            .includedMessage
                        : selectionCopy
                            .excludedMessage
                    }
                  </span>
                </div>

                <button
                  type="button"
                  aria-pressed={
                    includedInReport
                  }
                  disabled={
                    reportSelectionLoading
                  }
                  onClick={
                    () => {
                      void onToggleReportSelection();
                    }
                  }
                  style={{
                    flex:
                      "0 0 auto",

                    minHeight:
                      "34px",

                    padding:
                      "0 11px",

                    border:
                      includedInReport
                        ? "1px solid rgba(122, 203, 160, 0.30)"
                        : "1px solid rgba(126, 177, 255, 0.20)",

                    borderRadius:
                      "9px",

                    color:
                      includedInReport
                        ? "#a4dec2"
                        : "#b8c9df",

                    background:
                      includedInReport
                        ? "rgba(122, 203, 160, 0.055)"
                        : "rgba(126, 177, 255, 0.035)",

                    font:
                      "inherit",

                    fontSize:
                      "0.69rem",

                    fontWeight:
                      700,

                    cursor:
                      "pointer",
                  }}
                >
                  {
                    includedInReport
                      ? selectionCopy
                          .removeLabel
                      : selectionCopy
                          .addLabel
                  }
                </button>
              </div>
            )
          : null
      }

      <NativeRequestedAnalysisCard
        report={
          report
        }
        objective={
          objective
        }
      />
    </div>
  );
}

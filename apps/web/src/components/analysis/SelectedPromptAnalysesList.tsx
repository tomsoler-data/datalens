"use client";

import type {
  ReportPromptAnalysisView,
} from "./analysisTypes";


type SelectedPromptAnalysesListProps = {
  analyses: ReportPromptAnalysisView[];
  onRemoveAnalysis: (analysisId: string) => Promise<void>;
};


export default function SelectedPromptAnalysesList({
  analyses,
  onRemoveAnalysis,
}: SelectedPromptAnalysesListProps) {
  return (
<div
                                        style={{
                                          display:
                                            "grid",

                                          gap:
                                            "9px",

                                          marginTop:
                                            "14px",
                                        }}
                                      >
                                        {
                                          analyses.map(
                                            (
                                              selectedAnalysis,
                                              index
                                            ) => (
                                              <article
                                                key={
                                                  selectedAnalysis.id
                                                }
                                                style={{
                                                  display:
                                                    "grid",

                                                  gridTemplateColumns:
                                                    "minmax(0, 1fr) auto",

                                                  alignItems:
                                                    "center",

                                                  gap:
                                                    "14px",

                                                  padding:
                                                    "11px 12px",

                                                  border:
                                                    "1px solid rgba(154, 174, 204, 0.11)",

                                                  borderRadius:
                                                    "10px",

                                                  background:
                                                    "rgba(3, 8, 17, 0.34)",
                                                }}
                                              >
                                                <div
                                                  style={{
                                                    minWidth:
                                                      0,
                                                  }}
                                                >
                                                  <span
                                                    style={{
                                                      display:
                                                        "block",

                                                      color:
                                                        "#7fa18f",

                                                      fontSize:
                                                        "0.61rem",

                                                      fontWeight:
                                                        720,

                                                      letterSpacing:
                                                        "0.055em",

                                                      textTransform:
                                                        "uppercase",
                                                    }}
                                                  >
                                                    {
                                                      index +
                                                      1
                                                    }
                                                    {" · "}
                                                    {
                                                      selectedAnalysis
                                                        .source_label
                                                    }
                                                  </span>

                                                  <strong
                                                    style={{
                                                      display:
                                                        "block",

                                                      marginTop:
                                                        "4px",

                                                      overflow:
                                                        "hidden",

                                                      color:
                                                        "#dce6f3",

                                                      fontSize:
                                                        "0.75rem",

                                                      fontWeight:
                                                        620,

                                                      lineHeight:
                                                        1.45,

                                                      textOverflow:
                                                        "ellipsis",

                                                      whiteSpace:
                                                        "nowrap",
                                                    }}
                                                    title={
                                                      selectedAnalysis
                                                        .objective
                                                    }
                                                  >
                                                    {
                                                      selectedAnalysis
                                                        .objective
                                                    }
                                                  </strong>
                                                </div>

                                                <button
                                                  type="button"
                                                  onClick={
                                                    () =>
                                                      onRemoveAnalysis(
                                                        selectedAnalysis.id
                                                      )
                                                  }
                                                  style={{
                                                    flex:
                                                      "0 0 auto",

                                                    minHeight:
                                                      "32px",

                                                    padding:
                                                      "0 10px",

                                                    border:
                                                      "1px solid rgba(205, 142, 142, 0.20)",

                                                    borderRadius:
                                                      "8px",

                                                    color:
                                                      "#d7a6a6",

                                                    background:
                                                      "rgba(205, 142, 142, 0.035)",

                                                    font:
                                                      "inherit",

                                                    fontSize:
                                                      "0.66rem",

                                                    fontWeight:
                                                      700,

                                                    cursor:
                                                      "pointer",
                                                  }}
                                                  aria-label={
                                                    `Retirer du rapport : ${selectedAnalysis.objective}`
                                                  }
                                                >
                                                  Retirer
                                                </button>
                                              </article>
                                            )
                                          )
                                        }
                                      </div>
  );
}

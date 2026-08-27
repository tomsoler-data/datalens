import type { ReportAvailableAnalysisDetailView } from "../components/analysis/analysisTypes";

import { reportSourceLabel } from "../components/analysis/reportSelectionPresentation";
import styles from "./page.module.css";


type AvailableAnalysisSelectionHeaderProps = {
  analysis: ReportAvailableAnalysisDetailView;
  selected: boolean;
  reportSelectionLoading: boolean;
  setAvailableAnalysisReportSelection: (args: {
    analysis: ReportAvailableAnalysisDetailView;
    included: boolean;
  }) => Promise<void>;
};


export default function AvailableAnalysisSelectionHeader({
  analysis,
  selected,
  reportSelectionLoading,
  setAvailableAnalysisReportSelection,
}: AvailableAnalysisSelectionHeaderProps) {
  return (
<div
                                  style={{
                                    display:
                                      "flex",

                                    justifyContent:
                                      "space-between",

                                    alignItems:
                                      "flex-start",

                                    gap:
                                      "14px",

                                    flexWrap:
                                      "wrap",

                                    marginBottom:
                                      "14px",
                                  }}
                                >
                                  <div
                                    style={{
                                      minWidth:
                                        0,

                                      flex:
                                        "1 1 420px",
                                    }}
                                  >
                                    <span
                                      className={
                                        styles.eyebrow
                                      }
                                    >
                                      {
                                        reportSourceLabel(
                                          analysis
                                            .source_type
                                        )
                                      }
                                    </span>

                                    <h3
                                      style={{
                                        margin:
                                          "7px 0 0",
                                      }}
                                    >
                                      {
                                        analysis
                                          .objective ||
                                        "Analyse persistée"
                                      }
                                    </h3>

                                    <p
                                      className={
                                        styles.resultSubtitle
                                      }
                                      style={{
                                        marginBottom:
                                          0,
                                      }}
                                    >
                                      {
                                        selected
                                          ? "Incluse dans le rapport"
                                          : "Disponible"
                                      }
                                    </p>
                                  </div>


                                  <button
                                    type="button"
                                    aria-pressed={
                                      selected
                                    }
                                    disabled={
                                      reportSelectionLoading
                                    }
                                    onClick={
                                      () => {
                                        void setAvailableAnalysisReportSelection(
                                          {
                                            analysis,

                                            included:
                                              !selected,
                                          }
                                        );
                                      }
                                    }
                                    style={{
                                      minHeight:
                                        "34px",

                                      padding:
                                        "0 11px",

                                      border:
                                        selected
                                          ? "1px solid rgba(122, 203, 160, 0.28)"
                                          : "1px solid rgba(126, 177, 255, 0.22)",

                                      borderRadius:
                                        "9px",

                                      color:
                                        "inherit",

                                      background:
                                        selected
                                          ? "rgba(122, 203, 160, 0.08)"
                                          : "rgba(126, 177, 255, 0.05)",

                                      font:
                                        "inherit",

                                      fontSize:
                                        "0.7rem",

                                      fontWeight:
                                        700,

                                      cursor:
                                        reportSelectionLoading
                                          ? "wait"
                                          : "pointer",
                                    }}
                                  >
                                    {
                                      selected
                                        ? "Retirer du rapport"
                                        : "Ajouter au rapport"
                                    }
                                  </button>
                                </div>
  );
}

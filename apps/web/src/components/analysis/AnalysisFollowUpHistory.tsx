"use client";

import styles from "../../app/page.module.css";

import type {
  AnalysisFollowUpTurn,
} from "./analysisTypes";

import {
  nativePipelineHasExecutedResult,
} from "./analysisExecutionPresentation";


type AnalysisFollowUpHistoryProps = {
  turns: AnalysisFollowUpTurn[];
  reportSelectionLoading: boolean;
  onToggleReportSelection: (turnId: string) => Promise<void>;
};


export default function AnalysisFollowUpHistory({
  turns,
  reportSelectionLoading,
  onToggleReportSelection,
}: AnalysisFollowUpHistoryProps) {
  return (
<details
                                        className={
                                          styles.analysisFollowUpHistory
                                        }
                                      >
                                        <summary>
                                          Questions précédentes
                                          {" · "}
                                          {
                                            turns.length -
                                            1
                                          }
                                        </summary>

                                        <ol>
                                          {
                                            turns
                                              .slice(
                                                0,
                                                -1
                                              )
                                              .map(
                                                (
                                                  turn
                                                ) => (
                                                  <li
                                                    key={
                                                      turn.id
                                                    }
                                                    style={{
                                                      display:
                                                        "flex",

                                                      alignItems:
                                                        "center",

                                                      justifyContent:
                                                        "space-between",

                                                      gap:
                                                        "12px",
                                                    }}
                                                  >
                                                    <span>
                                                      {
                                                        turn.objective
                                                      }
                                                    </span>

                                                    {
                                                      nativePipelineHasExecutedResult(
                                                        turn.report
                                                      )
                                                        ? (
                                                            <button
                                                              type="button"
                                                              aria-pressed={
                                                                turn.included_in_report
                                                              }
                                                              disabled={
                                                                reportSelectionLoading
                                                              }
                                                              onClick={
                                                                () =>
                                                                  onToggleReportSelection(
                                                                    turn.id
                                                                  )
                                                              }
                                                              style={{
                                                                flex:
                                                                  "0 0 auto",

                                                                minHeight:
                                                                  "30px",

                                                                padding:
                                                                  "0 9px",

                                                                border:
                                                                  turn.included_in_report
                                                                    ? "1px solid rgba(122, 203, 160, 0.28)"
                                                                    : "1px solid rgba(126, 177, 255, 0.18)",

                                                                borderRadius:
                                                                  "8px",

                                                                color:
                                                                  turn.included_in_report
                                                                    ? "#a4dec2"
                                                                    : "#aebdd1",

                                                                background:
                                                                  turn.included_in_report
                                                                    ? "rgba(122, 203, 160, 0.05)"
                                                                    : "rgba(126, 177, 255, 0.03)",

                                                                font:
                                                                  "inherit",

                                                                fontSize:
                                                                  "0.64rem",

                                                                fontWeight:
                                                                  680,

                                                                cursor:
                                                                  "pointer",
                                                              }}
                                                            >
                                                              {
                                                                turn.included_in_report
                                                                  ? "Retirer"
                                                                  : "Ajouter au rapport"
                                                              }
                                                            </button>
                                                          )
                                                        : (
                                                            <small>
                                                              Non ajoutable
                                                            </small>
                                                          )
                                                    }
                                                  </li>
                                                )
                                              )
                                          }
                                        </ol>
                                      </details>
  );
}

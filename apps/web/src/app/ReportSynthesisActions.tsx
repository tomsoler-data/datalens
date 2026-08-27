import type { Dispatch, SetStateAction } from "react";

import type { ReportSelectionDetailsView } from "../components/analysis/analysisTypes";
import type { WorkspaceStep } from "../components/workspace/workspaceNavigationTypes";

import styles from "./page.module.css";


type ReportSynthesisActionsProps = {
  reportSelectionDetails: ReportSelectionDetailsView | null;
  pdfExportLoading: boolean;
  handlePdfExport: () => Promise<void>;
  setActiveStep: Dispatch<SetStateAction<WorkspaceStep>>;
};


export default function ReportSynthesisActions({
  reportSelectionDetails,
  pdfExportLoading,
  handlePdfExport,
  setActiveStep,
}: ReportSynthesisActionsProps) {
  return (
<div
                            className={
                              styles.submitArea
                            }
                          >
                            <div
                              className={
                                styles.submitInfo
                              }
                            >
                              <strong>
                                Rapport de synthèse
                              </strong>

                              <span>
                                Cette vue rassemble uniquement
                                les éléments de décision. Le PDF
                                est généré par l’API locale ; les
                                données brutes ne sont pas envoyées
                                vers un service externe.
                                {
                                  (
                                    reportSelectionDetails
                                      ?.selected_count ??
                                    0
                                  ) ===
                                  0
                                    ? " Sélectionnez au moins une analyse pour activer l’export."
                                    : ""
                                }
                              </span>
                            </div>

                            <div
                              style={{
                                display:
                                  "flex",

                                flexWrap:
                                  "wrap",

                                gap:
                                  "10px",

                                justifyContent:
                                  "flex-end",
                              }}
                            >
                              <button
                                className={
                                  styles.submitButton
                                }
                                type="button"
                                disabled={
                                  pdfExportLoading ||
                                  (
                                    reportSelectionDetails
                                      ?.selected_count ??
                                    0
                                  ) ===
                                  0
                                }
                                title={
                                  (
                                    reportSelectionDetails
                                      ?.selected_count ??
                                    0
                                  ) ===
                                  0
                                    ? "Sélectionnez au moins une analyse avant l’export."
                                    : undefined
                                }
                                onClick={
                                  handlePdfExport
                                }
                              >
                                {
                                  pdfExportLoading
                                    ? "Génération du PDF…"
                                    : "Exporter en PDF"
                                }
                              </button>


                              <button
                                className={
                                  styles.submitButton
                                }
                                type="button"
                                onClick={
                                  () =>
                                    setActiveStep(
                                      "analyses"
                                    )
                                }
                              >
                                Revoir les analyses
                              </button>
                            </div>
                          </div>
  );
}

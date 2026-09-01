import type { Dispatch, SetStateAction } from "react";

import type { ReportSelectionDetailsView } from "../components/analysis/analysisTypes";
import type { WorkspaceStep } from "../components/workspace/workspaceNavigationTypes";

import styles from "./page.module.css";


/*
 * DATALENS_REPORT_EXPORT_COMPACT_V0_1
 *
 * Export language is concise and product-facing.
 */


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
  const selectedCount =
    reportSelectionDetails
      ?.selected_count ??
    0;


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
          Export
        </strong>

        <span>
          {
            selectedCount ===
            0
              ? "Sélectionnez une analyse pour activer l'export."
              : "Le PDF est généré localement."
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
            selectedCount ===
            0
          }
          title={
            selectedCount ===
            0
              ? "Sélectionnez une analyse avant l'export."
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

import type { ComponentProps } from "react";

import SelectedPromptAnalysesList from "../components/analysis/SelectedPromptAnalysesList";
import styles from "./page.module.css";


/*
 * DATALENS_REPORT_COMPOSITION_COMPACT_V0_1
 *
 * Keep report composition operational while removing
 * explanatory prose that duplicates the surrounding workflow.
 */


type SelectedPromptAnalysesSectionProps = {
  selectedPromptAnalyses: ComponentProps<typeof SelectedPromptAnalysesList>["analyses"];
  removePromptAnalysisFromReport: ComponentProps<typeof SelectedPromptAnalysesList>["onRemoveAnalysis"];
  reportSelectionError: string | null;
  reportSelectionLoading: boolean;
};


export default function SelectedPromptAnalysesSection({
  selectedPromptAnalyses,
  removePromptAnalysisFromReport,
  reportSelectionError,
  reportSelectionLoading,
}: SelectedPromptAnalysesSectionProps) {
  return (
    <section
      aria-labelledby="selected-prompt-analyses-title"
      style={{
        marginTop:
          "14px",

        padding:
          "16px",

        border:
          "1px solid rgba(122, 203, 160, 0.16)",

        borderRadius:
          "14px",

        background:
          "rgba(4, 14, 19, 0.34)",
      }}
    >
      <div
        style={{
          display:
            "flex",

          alignItems:
            "flex-start",

          justifyContent:
            "space-between",

          gap:
            "16px",

          flexWrap:
            "wrap",
        }}
      >
        <div>
          <span
            className={
              styles.eyebrow
            }
          >
            Composition du rapport
          </span>


          <h2
            id="selected-prompt-analyses-title"
            style={{
              margin:
                "6px 0 0",

              color:
                "#eef4fc",

              fontSize:
                "1rem",

              fontWeight:
                600,
            }}
          >
            Analyses incluses
          </h2>


          <div
            style={{
              display:
                "flex",

              alignItems:
                "center",

              gap:
                "7px",

              marginTop:
                "7px",

              color:
                reportSelectionError
                  ? "#d7a6a6"
                  : "#7fa18f",

              fontSize:
                "0.68rem",

              fontWeight:
                650,
            }}
          >
            <span
              aria-hidden="true"
              style={{
                width:
                  "6px",

                height:
                  "6px",

                borderRadius:
                  "999px",

                background:
                  reportSelectionError
                    ? "#d7a6a6"
                    : reportSelectionLoading
                      ? "#8bb9ff"
                      : "#8cd7b7",
              }}
            />


            {
              reportSelectionError
                ? "Synchronisation indisponible"
                : reportSelectionLoading
                  ? "Synchronisation…"
                  : "Synchronisé"
            }
          </div>
        </div>


        <span
          style={{
            flex:
              "0 0 auto",

            padding:
              "6px 9px",

            border:
              "1px solid rgba(122, 203, 160, 0.22)",

            borderRadius:
              "999px",

            color:
              "#a4dec2",

            background:
              "rgba(122, 203, 160, 0.04)",

            fontSize:
              "0.67rem",

            fontWeight:
              700,
          }}
        >
          {
            selectedPromptAnalyses.length
          }
          {" sélectionnée"}
          {
            selectedPromptAnalyses.length >
            1
              ? "s"
              : ""
          }
        </span>
      </div>


      {
        selectedPromptAnalyses.length ===
        0
          ? (
              <div
                style={{
                  marginTop:
                    "13px",

                  padding:
                    "11px 13px",

                  border:
                    "1px dashed rgba(154, 174, 204, 0.14)",

                  borderRadius:
                    "10px",

                  color:
                    "#91a0b5",

                  fontSize:
                    "0.72rem",
                }}
              >
                Aucune analyse ajoutée.
              </div>
            )
          : (
              <SelectedPromptAnalysesList
                analyses={
                  selectedPromptAnalyses
                }
                onRemoveAnalysis={
                  removePromptAnalysisFromReport
                }
              />
            )
      }
    </section>
  );
}

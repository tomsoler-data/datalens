import type { ComponentProps } from "react";

import SelectedPromptAnalysesList from "../components/analysis/SelectedPromptAnalysesList";
import styles from "./page.module.css";


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
                                }}
                              >
                                <div>
                                  <span
                                    className={
                                      styles.eyebrow
                                    }
                                  >
                                    Sélection pour le rapport
                                  </span>

                                  <div
                                    style={{
                                      display:
                                        "flex",

                                      alignItems:
                                        "center",

                                      gap:
                                        "7px",

                                      marginBottom:
                                        "4px",

                                      color:
                                        reportSelectionError
                                          ? "#d7a6a6"
                                          : "#7fa18f",

                                      fontSize:
                                        "0.62rem",

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
                                        ? "Synchronisation du rapport indisponible"
                                        : reportSelectionLoading
                                          ? "Synchronisation…"
                                          : "Sélection synchronisée avec le serveur"
                                    }
                                  </div>

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
                                    Analyses ajoutées
                                  </h2>

                                  <p
                                    style={{
                                      margin:
                                        "6px 0 0",

                                      maxWidth:
                                        "680px",

                                      color:
                                        "#9eacc0",

                                      fontSize:
                                        "0.72rem",

                                      lineHeight:
                                        1.6,
                                    }}
                                  >
                                    Ces analyses seront reprises dans le rapport.
                                    Les retirer ici ne supprime pas leurs résultats :
                                    elles restent disponibles dans l’espace Analyses.
                                  </p>
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
                                            "14px",

                                          padding:
                                            "13px 14px",

                                          border:
                                            "1px dashed rgba(154, 174, 204, 0.14)",

                                          borderRadius:
                                            "10px",

                                          color:
                                            "#91a0b5",

                                          fontSize:
                                            "0.72rem",

                                          lineHeight:
                                            1.6,
                                        }}
                                      >
                                        Aucune analyse n’est encore ajoutée au rapport.
                                        Utilisez « Ajouter au rapport » sur une réponse
                                        exécutée pour la conserver.
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

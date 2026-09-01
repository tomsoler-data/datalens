import type { ChangeEvent, Dispatch, SetStateAction } from "react";

import type { MultiDatasetIngestion } from "./types";

import styles from "./page.module.css";


type DatasetImportControlsProps = {
  workflowDisplayName: string;
  setWorkflowDisplayName: Dispatch<SetStateAction<string>>;
  ingestion: MultiDatasetIngestion | null;
  ingestionLoading: boolean;
  preparationSessionLoading: boolean;
  handleDatasetsChange: (event: ChangeEvent<HTMLInputElement>) => Promise<void>;
};


export default function DatasetImportControls({
  workflowDisplayName,
  setWorkflowDisplayName,
  ingestion,
  ingestionLoading,
  preparationSessionLoading,
  handleDatasetsChange,
}: DatasetImportControlsProps) {
  return (
<>
                <div
                  style={{
                    display:
                      "grid",

                    gap:
                      "7px",

                    marginBottom:
                      "14px",
                  }}
                >
                  <label
                    htmlFor="workflow-display-name"
                    style={{
                      display:
                        "flex",

                      alignItems:
                        "baseline",

                      gap:
                        "7px",

                      fontSize:
                        "0.72rem",

                      fontWeight:
                        680,
                    }}
                  >
                    <span>
                      {
                        "Nom du workflow"
                      }
                    </span>

                    <span
                      style={{
                        fontSize:
                          "0.61rem",

                        fontWeight:
                          500,

                        opacity:
                          0.48,
                      }}
                    >
                      {
                        "(facultatif)"
                      }
                    </span>
                  </label>


                  <input
                    id="workflow-display-name"
                    type="text"
                    value={
                      workflowDisplayName
                    }
                    maxLength={
                      120
                    }
                    disabled={
                      ingestionLoading ||
                      preparationSessionLoading
                    }
                    placeholder={
                      "Ex. Analyse des ventes"
                    }
                    onChange={
                      (
                        event
                      ) => {
                        setWorkflowDisplayName(
                          event.target.value
                        );
                      }
                    }
                    style={{
                      width:
                        "100%",

                      minHeight:
                        "40px",

                      padding:
                        "0 12px",

                      border:
                        "1px solid rgba(255,255,255,0.09)",

                      borderRadius:
                        "10px",

                      outline:
                        "none",

                      background:
                        "rgba(255,255,255,0.018)",

                      color:
                        "inherit",

                      font:
                        "inherit",

                      fontSize:
                        "0.76rem",
                    }}
                  />


                  <span
                    style={{
                      fontSize:
                        "0.62rem",

                      lineHeight:
                        1.5,

                      opacity:
                        0.48,
                    }}
                  >
                    {
                      "Laissez vide pour laisser DataLens generer automatiquement un nom."
                    }
                  </span>
                </div>


                <label
                className={
                  `${styles.dropZone} ${
                    ingestion
                      ? styles.dropZoneCompact
                      : ""
                  }`
                }
              >
                <input
                  className={
                    styles.fileInput
                  }
                  type="file"
                  multiple
                  accept=".csv,text/csv"
                  onChange={
                    handleDatasetsChange
                  }
                />

                <strong
                  className={
                    styles.dropLabel
                  }
                >
                  {
                    ingestion
                      ? "Ajouter d’autres fichiers CSV"
                      : "Sélectionner un ou plusieurs fichiers CSV"
                  }
                </strong>

                <span
                  className={
                    styles.dropFormats
                  }
                >
                  CSV · plusieurs fichiers acceptés
                </span>

                <span
                  className={
                    styles.dropNote
                  }
                >
                  Les fichiers restent locaux. Plusieurs datasets peuvent
                  être analysés dans le même workspace.
                </span>
              </label>
              </>
  );
}

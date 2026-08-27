import type { ChangeEvent } from "react";

import type { WorkspaceStep } from "../components/workspace/workspaceNavigationTypes";

import { formatBytes } from "../components/preparation/preparationPresentation";
import styles from "./page.module.css";


type BusinessDocumentsSectionProps = {
  activeStep: WorkspaceStep;
  documents: File[];
  handleDocumentsChange: (event: ChangeEvent<HTMLInputElement>) => void;
};


export default function BusinessDocumentsSection({
  activeStep,
  documents,
  handleDocumentsChange,
}: BusinessDocumentsSectionProps) {
  return (
<section
              className={
                `${styles.panel} ${styles.contextPanel}`
              }
              style={{
                display:
                  activeStep ===
                    "documents"
                    ? undefined
                    : "none",
              }}
            >
              <div
                className={
                  styles.sectionHead
                }
              >
                <div>
                  <span
                    className={
                      styles.eyebrow
                    }
                  >
                    Contexte facultatif
                  </span>

                  <h2>
                    Documents métier
                  </h2>

                  <p
                    className={
                      styles.sectionDescription
                    }
                  >
                    Ajoutez des rapports, définitions, procédures ou briefs si
                    leur contenu peut aider à interpréter les résultats.
                  </p>
                </div>

                <span
                  className={
                    styles.optionalBadge
                  }
                >
                  Optionnel
                </span>
              </div>


              <label
                className={
                  `${styles.dropZone} ${
                    documents.length >
                      0
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
                  accept=".pdf,.doc,.docx,.txt,.md,text/plain,application/pdf"
                  onChange={
                    handleDocumentsChange
                  }
                />

                <strong
                  className={
                    styles.dropLabel
                  }
                >
                  {
                    documents.length >
                      0
                      ? "Ajouter d’autres documents"
                      : "Ajouter des documents métier"
                  }
                </strong>

                <span
                  className={
                    styles.dropFormats
                  }
                >
                  PDF · DOCX · TXT · MD
                </span>

                <span
                  className={
                    styles.dropNote
                  }
                >
                  Sélectionnez uniquement les documents utiles au contexte.
                  Ils ne remplacent jamais les calculs Python.
                </span>
              </label>


              {
                documents.length >
                0
                  ? (
                      <>
                        <div
                          className={
                            styles.fileList
                          }
                        >
                          {
                            documents.map(
                              (
                                file
                              ) => (
                                <div
                                  className={
                                    styles.fileRow
                                  }
                                  key={
                                    `${file.name}-${file.size}`
                                  }
                                >
                                  <span
                                    className={
                                      styles.fileBadge
                                    }
                                  >
                                    DOC
                                  </span>

                                  <div
                                    className={
                                      styles.fileMeta
                                    }
                                  >
                                    <strong>
                                      {
                                        file.name
                                      }
                                    </strong>

                                    <small>
                                      {
                                        formatBytes(
                                          file.size
                                        )
                                      }
                                    </small>
                                  </div>
                                </div>
                              )
                            )
                          }
                        </div>

                        <p
                          className={
                            styles.helper
                          }
                        >
                          Ces documents seront
                          utilisés uniquement pour
                          contextualiser les résultats
                          calculés par le moteur
                          analytique.
                        </p>
                      </>
                    )
                  : (
                      <p
                        className={
                          styles.helper
                        }
                      >
                        Optionnel. Sans document,
                        DataLens exécute uniquement
                        l’analyse déterministe.
                      </p>
                    )
              }
            </section>
  );
}

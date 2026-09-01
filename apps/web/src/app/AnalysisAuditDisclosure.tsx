import type { RoutedUnifiedAnalysisReportView } from "./workspaceAnalysisTypes";
import type { RagContextReport } from "./types";
import PrioritizationAuditPanel from "./PrioritizationAuditPanel";
import CompactFindingList from "../components/analysis/CompactFindingList";
import QualityList from "../components/analysis/QualityList";
import RagReportSummary from "../components/analysis/RagReportSummary";
import BlockedAnalysisList from "../components/analysis/BlockedAnalysisList";
import styles from "./page.module.css";


/*
 * DATALENS_COMPACT_ANALYSIS_AUDIT_V0_1
 *
 * Decision-facing audit summary first.
 * Existing detailed evidence remains intact behind
 * an explicit second-level disclosure.
 */


/*
 * DATALENS_NESTED_ANALYSIS_EVIDENCE_V0_1
 *
 * Heavy evidence families stay collapsed by default.
 * The user progressively opens only the proof needed.
 */


type AnalysisAuditDisclosureProps = {
  report: RoutedUnifiedAnalysisReportView;
  ragReport: RagContextReport | null;
};


export default function AnalysisAuditDisclosure({
  report,
  ragReport,
}: AnalysisAuditDisclosureProps) {
  const prioritizationAudit =
    report.prioritization_audit;


  const discoveredCount =
    prioritizationAudit
      ?.discovered_count
    ??
    report.inventory
      .discovered_analysis_count;


  const selectedCount =
    prioritizationAudit
      ?.selected_for_execution_count
    ??
    report.inventory
      .executed_analysis_count;


  const deferredCount =
    prioritizationAudit
      ?.deferred_count
    ??
    0;


  const rejectedCount =
    prioritizationAudit
      ?.rejected_count
    ??
    0;


  const nonExecutedCount =
    deferredCount +
    rejectedCount;


  return (
    <details
      className={
        `${styles.analysisDisclosure} ${styles.analysisTechnicalDisclosure}`
      }
    >
      <summary
        className={
          styles.analysisDisclosureSummary
        }
      >
        <div>
          <span
            className={
              styles.eyebrow
            }
          >
            Audit
          </span>

          <strong>
            Preuves & méthodologie
          </strong>

          <small>
            Contrôles, qualité et traçabilité disponibles à la demande.
          </small>
        </div>

        <span
          className={
            styles.analysisDisclosureMeta
          }
        >
          Détails
        </span>
      </summary>


      <div
        className={
          styles.analysisDisclosureBody
        }
      >
        <section
          className={
            styles.analysisAuditSummary
          }
          aria-label="Résumé de l'audit analytique"
        >
          <div
            className={
              styles.analysisAuditMetrics
            }
          >
            <article>
              <span>
                Découvertes
              </span>

              <strong>
                {
                  discoveredCount
                }
              </strong>
            </article>


            <article>
              <span>
                Retenues
              </span>

              <strong>
                {
                  selectedCount
                }
              </strong>
            </article>


            <article>
              <span>
                Différées
              </span>

              <strong>
                {
                  deferredCount
                }
              </strong>
            </article>


            <article>
              <span>
                Rejetées
              </span>

              <strong>
                {
                  rejectedCount
                }
              </strong>
            </article>
          </div>


          <div
            className={
              styles.analysisAuditSignal
            }
          >
            <span
              aria-hidden="true"
            >
              {"✓"}
            </span>

            <div>
              <strong>
                Sélection analytique contrôlée
              </strong>

              <small>
                {
                  nonExecutedCount ===
                  0
                    ? "Toutes les analyses retenues ont franchi les contrôles de priorisation."
                    : `${nonExecutedCount} analyse${nonExecutedCount > 1 ? "s" : ""} reste${nonExecutedCount > 1 ? "nt" : ""} documentée${nonExecutedCount > 1 ? "s" : ""} sans exécution automatique.`
                }
              </small>
            </div>
          </div>
        </section>


        <details
          className={
            styles.analysisAuditEvidence
          }
        >
          <summary>
            <span>
              Voir toutes les preuves
            </span>

            <small>
              Audit détaillé
            </small>
          </summary>


          <div
            className={
              styles.analysisAuditEvidenceBody
            }
          >
            {
              prioritizationAudit
                ? (
                    <details
                      className={
                        styles.analysisEvidenceSection
                      }
                    >
                      <summary>
                        <span>
                          Priorisation analytique
                        </span>

                        <small>
                          {
                            selectedCount
                          }
                          /
                          {
                            discoveredCount
                          }
                          {" retenues"}
                        </small>
                      </summary>

                      <div
                        className={
                          styles.analysisEvidenceSectionBody
                        }
                      >
                        <PrioritizationAuditPanel
                          audit={
                            prioritizationAudit
                          }
                        />
                      </div>
                    </details>
                  )
                : null
            }


            <CompactFindingList
              title="Analyses complémentaires"
              findings={
                report.additional_findings
              }
            />


            <CompactFindingList
              title="Diagnostics"
              findings={
                report.diagnostics
              }
            />


            <QualityList
              items={
                report.quality
              }
            />


            <CompactFindingList
              title="Analyses contextuelles"
              findings={
                report.context_analyses
              }
            />


            <BlockedAnalysisList
              items={
                report.blocked_analyses
              }
            />


            {
              ragReport
                ? (
                    <details
                      className={
                        styles.analysisEvidenceSection
                      }
                    >
                      <summary>
                        <span>
                          Contexte documentaire
                        </span>

                        <small>
                          {
                            ragReport.document_count
                          }
                          {
                            ragReport.document_count >
                            1
                              ? " documents"
                              : " document"
                          }
                        </small>
                      </summary>

                      <div
                        className={
                          styles.analysisEvidenceSectionBody
                        }
                      >
                        <RagReportSummary
                          rag={
                            ragReport
                          }
                        />
                      </div>
                    </details>
                  )
                : null
            }


            {
              report.methodology_notes.length >
              0
                ? (
                    <details
                      className={
                        styles.analysisEvidenceSection
                      }
                    >
                      <summary>
                        <span>
                          Méthodologie et traçabilité
                        </span>

                        <small>
                          {
                            report.methodology_notes.length
                          }
                          {
                            report.methodology_notes.length >
                            1
                              ? " notes"
                              : " note"
                          }
                        </small>
                      </summary>


                      <div
                        className={
                          styles.analysisEvidenceSectionBody
                        }
                      >
                        {/*
                         * DATALENS_METHODOLOGY_PRODUCT_SUMMARY_V0_1
                         *
                         * Product-facing methodology is separated
                         * from the raw technical evidence log.
                         */}

                        <section
                          className={
                            styles.methodologyProductSummary
                          }
                          aria-label="Résumé méthodologique"
                        >
                          <span
                            className={
                              styles.eyebrow
                            }
                          >
                            Résumé méthodologique
                          </span>


                          <div
                            className={
                              styles.methodologyPrinciples
                            }
                          >
                            <article>
                              <span
                                aria-hidden="true"
                              >
                                {"✓"}
                              </span>

                              <div>
                                <strong>
                                  Calculs déterministes
                                </strong>

                                <p>
                                  Les résultats numériques et les contraintes statistiques sont produits par le moteur analytique.
                                </p>
                              </div>
                            </article>


                            <article>
                              <span
                                aria-hidden="true"
                              >
                                {"✓"}
                              </span>

                              <div>
                                <strong>
                                  Qualité contrôlée
                                </strong>

                                <p>
                                  Les analyses conservent leurs contrôles de qualité, de cohérence et d&apos;exécutabilité.
                                </p>
                              </div>
                            </article>


                            <article>
                              <span
                                aria-hidden="true"
                              >
                                {"✓"}
                              </span>

                              <div>
                                <strong>
                                  Granularité préservée
                                </strong>

                                <p>
                                  DataLens n&apos;invente ni moyenne ni agrégation pour forcer une analyse ou supprimer une répétition de grain.
                                </p>
                              </div>
                            </article>


                            <article>
                              <span
                                aria-hidden="true"
                              >
                                {"✓"}
                              </span>

                              <div>
                                <strong>
                                  Jointures protégées
                                </strong>

                                <p>
                                  Les relations ambiguës restent bloquées tant qu&apos;un grain commun fiable ne peut pas être établi.
                                </p>
                              </div>
                            </article>


                            <article>
                              <span
                                aria-hidden="true"
                              >
                                {"✓"}
                              </span>

                              <div>
                                <strong>
                                  Résultats traçables
                                </strong>

                                <p>
                                  Les règles, vues dérivées et décisions techniques restent disponibles pour audit.
                                </p>
                              </div>
                            </article>
                          </div>
                        </section>


                        <details
                          className={
                            styles.methodologyTechnicalNotes
                          }
                        >
                          <summary>
                            <span>
                              Notes techniques détaillées
                            </span>

                            <small>
                              {
                                report.methodology_notes.length
                              }
                              {
                                report.methodology_notes.length >
                                1
                                  ? " notes"
                                  : " note"
                              }
                            </small>
                          </summary>


                          <div
                            className={
                              `${styles.analysisEvidenceMethodology} ${styles.methodologyTechnicalNotesBody}`
                            }
                          >
                            {
                              report.methodology_notes.map(
                                (
                                  note
                                ) => (
                                  <p
                                    key={
                                      note
                                    }
                                  >
                                    {
                                      note
                                    }
                                  </p>
                                )
                              )
                            }
                          </div>
                        </details>
                      </div>
                    </details>
                  )
                : null
            }
          </div>
        </details>
      </div>
    </details>
  );
}

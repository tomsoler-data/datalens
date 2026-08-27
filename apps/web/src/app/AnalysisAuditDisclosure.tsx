import type { RoutedUnifiedAnalysisReportView } from "./workspaceAnalysisTypes";
import type { RagContextReport } from "./types";
import PrioritizationAuditPanel from "./PrioritizationAuditPanel";
import CompactFindingList from "../components/analysis/CompactFindingList";
import QualityList from "../components/analysis/QualityList";
import RagReportSummary from "../components/analysis/RagReportSummary";
import BlockedAnalysisList from "../components/analysis/BlockedAnalysisList";
import styles from "./page.module.css";


type AnalysisAuditDisclosureProps = {
  report: RoutedUnifiedAnalysisReportView;
  ragReport: RagContextReport | null;
};


export default function AnalysisAuditDisclosure({
  report,
  ragReport,
}: AnalysisAuditDisclosureProps) {
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
                                    Diagnostics, qualité, RAG, analyses non exécutées
                                    et traçabilité du moteur.
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
                                {
                                  report.prioritization_audit
                                    ? (
                                        <PrioritizationAuditPanel
                                          audit={
                                            report.prioritization_audit
                                          }
                                        />
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
                                        <RagReportSummary
                                          rag={
                                            ragReport
                                          }
                                        />
                                      )
                                    : null
                                }

                                {
                                  report.methodology_notes.length >
                                  0
                                    ? (
                                        <details
                                          className={
                                            styles.technicalPanel
                                          }
                                        >
                                          <summary>
                                            Méthodologie et traçabilité
                                          </summary>

                                          <div
                                            className={
                                              styles.technicalReasons
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
                                      )
                                    : null
                                }
                              </div>
                            </details>
  );
}

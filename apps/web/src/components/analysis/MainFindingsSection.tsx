"use client";

import type {
  FindingRagContext,
  ReportFinding,
} from "../../app/types";

import CompactFindingList
  from "./CompactFindingList";

import FindingCard
  from "./FindingCard";

import styles
  from "../../app/page.module.css";


type MainFindingsSectionProps = {
  findings: ReportFinding[];
  ragContextByAnalysisId: Map<string, FindingRagContext>;
};


export default function MainFindingsSection({
  findings,
  ragContextByAnalysisId,
}: MainFindingsSectionProps) {
  return (
<details
                                      className={
                                        styles.analysisDisclosure
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
                                            Exploration automatique
                                          </span>

                                          <strong>
                                            Analyses complémentaires
                                          </strong>

                                          <small>
                                            Autres signaux utiles découverts par DataLens.
                                            Ils restent séparés de la réponse à votre demande.
                                          </small>
                                        </div>

                                        <span
                                          className={
                                            styles.analysisDisclosureCount
                                          }
                                        >
                                          {
                                            findings.length
                                          }
                                        </span>
                                      </summary>

                                      <div
                                        className={
                                          styles.analysisDisclosureBody
                                        }
                                      >
                                        <div
                                          className={
                                            styles.explanationGrid
                                          }
                                        >
                                          {
                                            findings
                                              .slice(
                                                0,
                                                3
                                              )
                                              .map(
                                                (
                                                  finding,
                                                  index
                                                ) => (
                                                  <FindingCard
                                                    finding={
                                                      finding
                                                    }
                                                    index={
                                                      index
                                                    }
                                                    ragContext={
                                                      finding.analysis_id
                                                        ? (
                                                            ragContextByAnalysisId.get(
                                                              finding.analysis_id
                                                            ) ??
                                                            null
                                                          )
                                                        : null
                                                    }
                                                    key={
                                                      `${
                                                        finding.analysis_id ??
                                                        `${finding.family}-${finding.title}`
                                                      }-${index}`
                                                    }
                                                  />
                                                )
                                              )
                                          }
                                        </div>

                                        {
                                          findings.length >
                                          3
                                            ? (
                                                <CompactFindingList
                                                  title="Autres analyses découvertes"
                                                  findings={
                                                    findings.slice(
                                                      3
                                                    )
                                                  }
                                                />
                                              )
                                            : null
                                        }
                                      </div>
                                    </details>
  );
}

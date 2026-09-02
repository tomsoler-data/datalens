import type {
  AINativePipelineReportView,
} from "./analysisTypes";

import {
  familyLabel,
  plannerEngineLabel,
} from "./analysisPlanningPresentation";

import {
  formatDecimal,
  formatNumber,
  formatPercent,
  formatTemporalDisplayValue,
  friendlyVariableLabel,
  metricNumber,
  metricString,
} from "./analysisPresentation";

import ExpandableChart from "./charts/ExpandableChart";
import LineBandChart from "./charts/LineBandChart";
import NativeHistogramChart from "./charts/NativeHistogramChart";
import RequestedBarChart from "./charts/RequestedBarChart";
import RequestedBoxPlotChart from "./charts/RequestedBoxPlotChart";
import RequestedHeatmapChart from "./charts/RequestedHeatmapChart";
import RequestedLorenzChart from "./charts/RequestedLorenzChart";
import ScatterPlot from "./charts/ScatterPlot";
import SimpleLineChart from "./charts/SimpleLineChart";

import styles from "../../app/page.module.css";


function NativeRequestedAnalysisSingleCard({
  report,
  objective,
}: {
  report:
    AINativePipelineReportView;

  objective:
    string;
}) {
  const item =
    report.items.find(
      (
        candidate
      ) =>
        candidate.pipeline_status ===
        "executed" &&
        candidate.native_tool
          ?.execution
          ?.result
          !==
          null
    ) ??
    null;


  const nativeTool =
    item
      ?.native_tool ??
    null;


  const execution =
    nativeTool
      ?.execution ??
    null;


  const result =
    execution
      ?.result ??
    null;


  if (
    !item ||
    !nativeTool ||
    !execution ||
    !result
  ) {
    const blockedItem =
      report.planner.items.find(
        (
          candidate
        ) =>
          candidate.validation_status ===
            "blocked" ||
          candidate.validation_status ===
            "rejected" ||
          candidate.validation_status ===
            "ambiguous"
      ) ??
      null;


    if (
      !blockedItem
    ) {
      return null;
    }


    const contract =
      blockedItem.contract;


    const statusLabel =
      blockedItem.validation_status ===
        "blocked"
        ? "Analyse bloquée"
        : blockedItem.validation_status ===
            "rejected"
          ? "Plan rejeté"
          : "Plan ambigu";


    const reasons =
      [
        ...(
          contract
            ?.blockers ??
          []
        ),
        ...blockedItem.errors,
        ...(
          blockedItem.proposal
            .blockers ??
          []
        ),
      ].filter(
        (
          reason,
          index,
          values
        ) =>
          Boolean(
            reason
          ) &&
          values.indexOf(
            reason
          ) ===
            index
      );


    return (
      <section
        aria-labelledby="requested-ai-analysis-blocked-title"
        style={{
          marginBottom:
            "26px",

          padding:
            "18px",

          border:
            "1px solid rgba(255, 178, 92, 0.24)",

          borderRadius:
            "16px",

          background:
            "linear-gradient(180deg, rgba(139, 83, 22, 0.10), rgba(10, 18, 32, 0.20))",
        }}
      >
        <div
          className={
            styles.sectionHead
          }
          style={{
            marginBottom:
              "14px",
          }}
        >
          <div>
            <span
              className={
                styles.eyebrow
              }
            >
              Analyse demandée
            </span>

            <h2
              id="requested-ai-analysis-blocked-title"
            >
              {
                blockedItem.proposal.title ||
                objective.trim() ||
                "Analyse demandée"
              }
            </h2>

            <p
              className={
                styles.resultSubtitle
              }
            >
              {
                objective.trim() ||
                report.planner.objective
              }
            </p>
          </div>


          <div
            style={{
              display:
                "flex",

              gap:
                "7px",

              flexWrap:
                "wrap",

              justifyContent:
                "flex-end",
            }}
          >
            <span
              style={{
                padding:
                  "6px 9px",

                border:
                  "1px solid rgba(255, 178, 92, 0.28)",

                borderRadius:
                  "999px",

                fontSize:
                  "0.69rem",

                fontWeight:
                  700,
              }}
            >
              {
                familyLabel(
                  blockedItem.proposal.family
                )
              }
            </span>

            <span
              style={{
                padding:
                  "6px 9px",

                border:
                  "1px solid rgba(255, 178, 92, 0.32)",

                borderRadius:
                  "999px",

                fontSize:
                  "0.69rem",

                fontWeight:
                  700,
              }}
            >
              {
                statusLabel
              }
            </span>
          </div>
        </div>


        <div
          className={
            styles.technicalReasons
          }
          style={{
            marginBottom:
              "12px",
          }}
        >
          <strong>
            DataLens n’exécute pas cette analyse
          </strong>

          <p>
            Le plan n’a pas franchi la validation
            déterministe. Aucun outil natif ni calcul
            statistique n’a été lancé pour cette demande.
          </p>

          {
            reasons.length >
            0
              ? (
                  reasons.map(
                    (
                      reason,
                      reasonIndex
                    ) => (
                      <p
                        key={
                          `${reasonIndex}-${reason}`
                        }
                      >
                        {
                          reason
                        }
                      </p>
                    )
                )
              )
            : (
                <p>
                  Le planner n’a pas produit de contrat
                  exécutable pour cet objectif.
                </p>
              )
          }
        </div>


        <div
          className={
            styles.evidenceFlow
          }
        >
          <article
            className={
              styles.evidenceItem
            }
          >
            <span>
              Planner
            </span>

            <strong>
              {
                plannerEngineLabel(
                  report.planner_model
                )
              }
            </strong>

            <small>
              {
                report.planner_model
              }
              {" · "}
              {
                report.planner
                  .attempt_count ??
                1
              }
              {" tentative(s)"}
            </small>
          </article>


          <article
            className={
              styles.evidenceItem
            }
          >
            <span>
              Validation
            </span>

            <strong>
              Python
            </strong>

            <small>
              {
                statusLabel
              }
            </small>
          </article>


          <article
            className={
              styles.evidenceItem
            }
          >
            <span>
              Tool calling
            </span>

            <strong>
              Non exécuté
            </strong>

            <small>
              Qwen n’est pas autorisé à poursuivre
            </small>
          </article>


          <article
            className={
              styles.evidenceItem
            }
          >
            <span>
              Calcul
            </span>

            <strong>
              Non exécuté
            </strong>

            <small>
              Aucun résultat statistique produit
            </small>
          </article>
        </div>


        {
          report.planner.retry_triggered
            ? (
                <div
                  className={
                    styles.technicalReasons
                  }
                  style={{
                    marginTop:
                      "12px",
                  }}
                >
                  <p>
                    Le planner a reçu un retry contrôlé après
                    le rejet initial. Python a conservé le
                    garde-fou et aucune substitution silencieuse
                    n’a été exécutée.
                  </p>
                </div>
              )
            : null
        }
      </section>
    );
  }


  const variables =
    execution
      .arguments
      .variables ??
    {};


  const chartData =
    result.chart_data ??
    [];


  const xColumn =
    variables.x ??
    null;


  const yColumn =
    variables.y ??
    null;


  const groupColumn =
    variables.group ??
    null;


  const productColumn =
    variables.product ??
    null;


  const dimensionColumn =
    productColumn ??
    variables.dimension ??
    groupColumn ??
    null;


  const valueColumn =
    variables.value ??
    null;


  const timeColumn =
    variables.time ??
    null;


  const statisticalResult =
    result.statistical_result ??
    {};


  const count =
    metricNumber(
      result.metrics,
      "count"
    ) ??
    metricNumber(
      result.metrics,
      "valid_observations"
    ) ??
    metricNumber(
      result.metrics,
      "valid_pairs"
    ) ??
    (
      typeof statisticalResult.n ===
        "number"
        ? statisticalResult.n
        : null
    );


  const mean =
    metricNumber(
      result.metrics,
      "mean"
    );


  const median =
    metricNumber(
      result.metrics,
      "median"
    );


  const standardDeviation =
    metricNumber(
      result.metrics,
      "std"
    );


  const coefficient =
    metricNumber(
      result.metrics,
      "coefficient"
    ) ??
    (
      typeof statisticalResult.coefficient ===
        "number"
        ? statisticalResult.coefficient
        : null
    );


  const pValue =
    metricNumber(
      result.metrics,
      "p_value"
    ) ??
    (
      typeof statisticalResult.p_value ===
        "number"
        ? statisticalResult.p_value
        : null
    );


  const descriptivePearson =
    metricNumber(
      result.metrics,
      "pearson_r"
    );


  const descriptiveSpearman =
    metricNumber(
      result.metrics,
      "spearman_rho"
    );


  const cramersV =
    typeof statisticalResult.cramers_v ===
      "number"
      ? statisticalResult.cramers_v
      : null;


  const groupCount =
    metricNumber(
      result.metrics,
      "group_count"
    );


  const resultCount =
    metricNumber(
      result.metrics,
      "result_count"
    );


  const aggregationFunction =
    metricString(
      result.metrics,
      "aggregation_function"
    );


  const periodCount =
    metricNumber(
      result.metrics,
      "period_count"
    );


  const timeStart =
    result.metrics[
      "time_start"
    ];


  const timeEnd =
    result.metrics[
      "time_end"
    ];


  const periodMedianMin =
    metricNumber(
      result.metrics,
      "period_median_min"
    );


  const periodMedianMax =
    metricNumber(
      result.metrics,
      "period_median_max"
    );


  const periodValueMin =
    metricNumber(
      result.metrics,
      "period_value_min"
    );


  const periodValueMax =
    metricNumber(
      result.metrics,
      "period_value_max"
    );


  const q1 =
    metricNumber(
      result.metrics,
      "q1"
    );


  const q3 =
    metricNumber(
      result.metrics,
      "q3"
    );


  const iqr =
    metricNumber(
      result.metrics,
      "iqr"
    );


  const outlierCountIqr =
    metricNumber(
      result.metrics,
      "outlier_count_iqr"
    );


  const outlierRatioIqr =
    metricNumber(
      result.metrics,
      "outlier_ratio_iqr"
    );


  const isOutlierRequest =
    result.family ===
      "distribution" &&
    /outlier|atypiqu|aberrant/i.test(
      objective
    );


  const lowerIqrBound =
    q1 !==
      null &&
    iqr !==
      null
      ? q1 -
        1.5 *
        iqr
      : null;


  const upperIqrBound =
    q3 !==
      null &&
    iqr !==
      null
      ? q3 +
        1.5 *
        iqr
      : null;


  const requestedVariableLabel =
    friendlyVariableLabel(
      valueColumn ??
      metricString(
        result.metrics,
        "column"
      ) ??
      "Valeur"
    );


  const requestedResultTitle =
    isOutlierRequest
      ? `Valeurs atypiques · ${requestedVariableLabel}`
      : (
          objective.trim() ||
          report.planner.objective.trim() ||
          result.title
        );


  const requestedAnalysisTitleId =
    `requested-ai-analysis-${item.contract_id.replace(
      /[^a-zA-Z0-9_-]/g,
      "-"
    )}`;


  const kpis:
    {
      label:
        string;

      value:
        string;
    }[] = [];


  if (
    isOutlierRequest
  ) {
    if (
      count !==
      null
    ) {
      kpis.push(
        {
          label:
            "Valeurs analysées",

          value:
            formatNumber(
              count
            ),
        }
      );
    }


    if (
      outlierCountIqr !==
      null
    ) {
      kpis.push(
        {
          label:
            "Outliers détectés",

          value:
            formatNumber(
              outlierCountIqr
            ),
        }
      );
    }


    if (
      outlierRatioIqr !==
      null
    ) {
      kpis.push(
        {
          label:
            "Part des observations",

          value:
            formatPercent(
              outlierRatioIqr
            ),
        }
      );
    }


    kpis.push(
      {
        label:
          "Méthode",

        value:
          "IQR · 1,5×",
      }
    );
  }


  else if (
    count !==
    null
  ) {
    kpis.push(
      {
        label:
          result.family ===
            "time_series"
            ? "Points temporels"
            : "Observations",

        value:
          formatNumber(
            count
          ),
      }
    );
  }


  if (
    result.family ===
      "time_series" &&
    periodCount !==
      null
  ) {
    kpis.push(
      {
        label:
          "Périodes",

        value:
          formatNumber(
            periodCount
          ),
      }
    );
  }


  if (
    result.family ===
      "time_series" &&
    timeStart !==
      undefined &&
    timeStart !==
      null
  ) {
    kpis.push(
      {
        label:
          "Début",

        value:
          formatTemporalDisplayValue(
            timeStart
          ),
      }
    );
  }


  if (
    result.family ===
      "time_series" &&
    timeEnd !==
      undefined &&
    timeEnd !==
      null
  ) {
    kpis.push(
      {
        label:
          "Fin",

        value:
          formatTemporalDisplayValue(
            timeEnd
          ),
      }
    );
  }


  if (
    mean !==
    null
  ) {
    kpis.push(
      {
        label:
          "Moyenne",

        value:
          formatDecimal(
            mean
          ),
      }
    );
  }


  if (
    median !==
    null
  ) {
    kpis.push(
      {
        label:
          "Médiane",

        value:
          formatDecimal(
            median
          ),
      }
    );
  }


  if (
    standardDeviation !==
    null
  ) {
    kpis.push(
      {
        label:
          "Écart-type",

        value:
          formatDecimal(
            standardDeviation
          ),
      }
    );
  }


  if (
    coefficient !==
    null
  ) {
    kpis.push(
      {
        label:
          "Coefficient",

        value:
          formatDecimal(
            coefficient
          ),
      }
    );
  }


  if (
    descriptivePearson !==
    null
  ) {
    kpis.push(
      {
        label:
          "Pearson r (descriptif)",

        value:
          formatDecimal(
            descriptivePearson
          ),
      }
    );
  }


  if (
    descriptiveSpearman !==
    null
  ) {
    kpis.push(
      {
        label:
          "Spearman rho (descriptif)",

        value:
          formatDecimal(
            descriptiveSpearman
          ),
      }
    );
  }


  if (
    cramersV !==
    null
  ) {
    kpis.push(
      {
        label:
          "V de Cramér",

        value:
          formatDecimal(
            cramersV
          ),
      }
    );
  }


  if (
    pValue !==
    null
  ) {
    kpis.push(
      {
        label:
          "p-value",

        value:
          formatDecimal(
            pValue
          ),
      }
    );
  }


  if (
    groupCount !==
    null
  ) {
    kpis.push(
      {
        label:
          "Groupes",

        value:
          formatNumber(
            groupCount
          ),
      }
    );
  }


  if (
    result.family ===
      "ranking" &&
    resultCount !==
      null
  ) {
    kpis.push(
      {
        label:
          "Résultats classés",

        value:
          formatNumber(
            resultCount
          ),
      }
    );
  }


  if (
    aggregationFunction !==
      null &&
    (
      result.family ===
        "aggregation" ||
      result.family ===
        "ranking" ||
      result.family ===
        "time_series"
    )
  ) {
    const aggregationLabels:
      Record<
        string,
        string
      > = {
        sum:
          "Somme",

        mean:
          "Moyenne",

        median:
          "Médiane",

        min:
          "Minimum",

        max:
          "Maximum",

        count:
          "Comptage",

        distinct_count:
          "Comptage distinct",
      };


    kpis.push(
      {
        label:
          "Agrégation",

        value:
          aggregationLabels[
            aggregationFunction
          ] ??
          aggregationFunction,
      }
    );
  }


  if (
    result.family ===
      "time_series" &&
    periodMedianMin !==
      null
  ) {
    kpis.push(
      {
        label:
          "Médiane min.",

        value:
          formatDecimal(
            periodMedianMin
          ),
      }
    );
  }


  if (
    result.family ===
      "time_series" &&
    periodMedianMax !==
      null
  ) {
    kpis.push(
      {
        label:
          "Médiane max.",

        value:
          formatDecimal(
            periodMedianMax
          ),
      }
    );
  }


  if (
    result.family ===
      "time_series" &&
    aggregationFunction !==
      "median" &&
    periodValueMin !==
      null
  ) {
    kpis.push(
      {
        label:
          "Valeur min.",

        value:
          formatDecimal(
            periodValueMin
          ),
      }
    );
  }


  if (
    result.family ===
      "time_series" &&
    aggregationFunction !==
      "median" &&
    periodValueMax !==
      null
  ) {
    kpis.push(
      {
        label:
          "Valeur max.",

        value:
          formatDecimal(
            periodValueMax
          ),
      }
    );
  }


  return (
    <section
      aria-labelledby={
        requestedAnalysisTitleId
      }
      className={
        styles.requestedAnswerCard
      }
    >
      <div
        className={
          styles.sectionHead
        }
        style={{
          marginBottom:
            "14px",
        }}
      >
        <div>
          <span
            className={
              styles.eyebrow
            }
          >
            Résultat vérifié
          </span>

          <h2
            id={
              requestedAnalysisTitleId
            }
          >
            {
              requestedResultTitle
            }
          </h2>

          <p
            className={
              styles.resultSubtitle
            }
          >
            {
              familyLabel(
                result.family
              )
            }
            {" · "}
            {
              result.dataset_filename ??
              "Dataset analysé"
            }
          </p>
        </div>


        <div
          style={{
            display:
              "flex",

            gap:
              "7px",

            flexWrap:
              "wrap",

            justifyContent:
              "flex-end",
          }}
        >
          <span
            style={{
              padding:
                "6px 9px",

              border:
                "1px solid rgba(126, 177, 255, 0.18)",

              borderRadius:
                "999px",

              fontSize:
                "0.69rem",

              fontWeight:
                700,
            }}
          >
            {
              familyLabel(
                result.family
              )
            }
          </span>

          <span
            style={{
              padding:
                "6px 9px",

              border:
                "1px solid rgba(122, 203, 160, 0.25)",

              borderRadius:
                "999px",

              fontSize:
                "0.69rem",

              fontWeight:
                700,
            }}
          >
            Exécution vérifiée
          </span>
        </div>
      </div>


      {
        !isOutlierRequest &&
        Object.keys(
          variables
        ).length >
        0
          ? (
              <div
                style={{
                  display:
                    "flex",

                  gap:
                    "7px",

                  flexWrap:
                    "wrap",

                  marginBottom:
                    "12px",
                }}
              >
                {
                  Object.entries(
                    variables
                  ).map(
                    (
                      [
                        role,
                        column,
                      ]
                    ) => (
                      <span
                        key={
                          `${role}-${column}`
                        }
                        style={{
                          padding:
                            "6px 8px",

                          border:
                            "1px solid rgba(255,255,255,0.07)",

                          borderRadius:
                            "8px",

                          fontSize:
                            "0.72rem",
                        }}
                      >
                        <strong>
                          {
                            role
                          }
                        </strong>

                        {" · "}

                        {
                          friendlyVariableLabel(
                            column
                          )
                        }
                      </span>
                    )
                  )
                }
              </div>
            )
          : null
      }


      {
        !isOutlierRequest &&
        result.summary.length >
        0
          ? (
              <div
                className={
                  styles.technicalReasons
                }
                style={{
                  marginBottom:
                    "12px",
                }}
              >
                {
                  result.summary
                    .slice(
                      0,
                      3
                    )
                    .map(
                      (
                        summary
                      ) => (
                        <p
                          key={
                            summary
                          }
                        >
                          {
                            summary
                          }
                        </p>
                      )
                    )
                }
              </div>
            )
          : null
      }


      {
        kpis.length >
        0
          ? (
              <div
                className={
                  styles.metricsGrid
                }
                style={{
                  marginBottom:
                    "14px",
                }}
              >
                {
                  kpis
                    .slice(
                      0,
                      4
                    )
                    .map(
                      (
                        kpi
                      ) => (
                        <article
                          className={
                            styles.metricCard
                          }
                          key={
                            kpi.label
                          }
                        >
                          <span>
                            {
                              kpi.label
                            }
                          </span>

                          <strong>
                            {
                              kpi.value
                            }
                          </strong>
                        </article>
                      )
                    )
                }
              </div>
            )
          : null
      }


      {
        isOutlierRequest
          ? (
              <section
                className={
                  styles.requestedAnswerStatement
                }
              >
                <span>
                  Conclusion
                </span>

                <strong>
                  {
                    outlierCountIqr ===
                      null
                      ? "Le calcul des valeurs atypiques est disponible dans les preuves."
                      : outlierCountIqr ===
                          0
                        ? "Aucune valeur atypique détectée avec la règle IQR 1,5×."
                        : `${formatNumber(
                            outlierCountIqr
                          )} valeur${
                            outlierCountIqr >
                            1
                              ? "s"
                              : ""
                          } atypique${
                            outlierCountIqr >
                            1
                              ? "s"
                              : ""
                          } détectée${
                            outlierCountIqr >
                            1
                              ? "s"
                              : ""
                          } sur ${requestedVariableLabel.toLowerCase()}.`
                  }
                </strong>

                <p>
                  Les valeurs sont signalées, pas supprimées.
                  DataLens conserve les données et laisse la décision
                  de traitement à l’analyste.
                </p>
              </section>
            )
          : null
      }


      {
        isOutlierRequest &&
        q1 !==
          null &&
        q3 !==
          null &&
        iqr !==
          null
          ? (
              <div
                className={
                  styles.outlierEvidenceGrid
                }
              >
                {[
                  {
                    label:
                      "Q1",

                    value:
                      formatDecimal(
                        q1
                      ),
                  },

                  {
                    label:
                      "Q3",

                    value:
                      formatDecimal(
                        q3
                      ),
                  },

                  {
                    label:
                      "IQR",

                    value:
                      formatDecimal(
                        iqr
                      ),
                  },

                  {
                    label:
                      "Borne basse",

                    value:
                      lowerIqrBound !==
                        null
                        ? formatDecimal(
                            lowerIqrBound
                          )
                        : "—",
                  },

                  {
                    label:
                      "Borne haute",

                    value:
                      upperIqrBound !==
                        null
                        ? formatDecimal(
                            upperIqrBound
                          )
                        : "—",
                  },
                ].map(
                  (
                    evidence
                  ) => (
                    <article
                      className={
                        styles.outlierEvidenceItem
                      }
                      key={
                        evidence.label
                      }
                    >
                      <span>
                        {
                          evidence.label
                        }
                      </span>

                      <strong>
                        {
                          evidence.value
                        }
                      </strong>
                    </article>
                  )
                )}
              </div>
            )
          : null
      }


      {
        result.chart_type ===
          "line" &&
        chartData.length >
          0
          ? (
              <div
                className={
                  styles.chartPanel
                }
              >
                <div
                  className={
                    styles.chartHeader
                  }
                >
                  <div>
                    <h3>
                      {
                        result.family ===
                          "time_series"
                          ? requestedResultTitle
                          : result.title
                      }
                    </h3>

                    <p>
                      {
                        result.family ===
                          "time_series"
                          ? (
                              `Évolution de ${friendlyVariableLabel(
                                valueColumn ??
                                "Valeur"
                              )} selon ${friendlyVariableLabel(
                                timeColumn ??
                                "Période"
                              )} · calcul déterministe.`
                            )
                          : "Évolution calculée par le moteur Python déterministe."
                      }
                    </p>
                  </div>
                </div>

                <ExpandableChart
                  title={
                    requestedResultTitle
                  }
                >
                  <SimpleLineChart
                    data={
                      chartData
                    }
                    xLabel={
                      friendlyVariableLabel(
                        timeColumn ??
                        "Période"
                      )
                    }
                    yLabel={
                      friendlyVariableLabel(
                        valueColumn ??
                        "Valeur"
                      )
                    }
                  />
                </ExpandableChart>
              </div>
            )
          : null
      }


      {
        result.chart_type ===
          "line_band" &&
        chartData.length >
          0
          ? (
              <div
                className={
                  styles.chartPanel
                }
              >
                <div
                  className={
                    styles.chartHeader
                  }
                >
                  <div>
                    <h3>
                      {
                        `Évolution de ${
                          friendlyVariableLabel(
                            valueColumn ??
                            "Valeur"
                          )
                        } selon ${
                          friendlyVariableLabel(
                            timeColumn ??
                            "Temps"
                          )
                        }`
                      }
                    </h3>

                    <p>
                      Médiane par période · bande
                      interquartile Q1–Q3 · calcul
                      déterministe.
                    </p>
                  </div>
                </div>

                <ExpandableChart
                  title={
                    result.title
                  }
                >
                  <LineBandChart
                    data={
                      chartData
                    }
                    yLabel={
                      friendlyVariableLabel(
                        valueColumn ??
                        "Valeur"
                      )
                    }
                  />
                </ExpandableChart>
              </div>
            )
          : null
      }


      {
        result.chart_type ===
          "histogram" &&
        chartData.length >
          0
          ? (
              <div
                className={
                  styles.chartPanel
                }
              >
                <div
                  className={
                    styles.chartHeader
                  }
                >
                  <div>
                    <h3>
                      {
                        isOutlierRequest
                          ? "Distribution et bornes IQR"
                          : "Distribution observée"
                      }
                    </h3>

                    <p>
                      {
                        isOutlierRequest
                          ? (
                              "Les zones extérieures aux bornes IQR 1,5× " +
                              "correspondent au périmètre des valeurs atypiques."
                            )
                          : "Histogramme calculé par le moteur déterministe."
                      }
                    </p>
                  </div>
                </div>

                <ExpandableChart
                  title={
                    result.title
                  }
                >
                  <NativeHistogramChart
                    data={
                      chartData
                    }
                    valueLabel={
                      requestedVariableLabel
                    }
                    lowerBound={
                      lowerIqrBound
                    }
                    upperBound={
                      upperIqrBound
                    }
                    highlightOutliers={
                      isOutlierRequest
                    }
                  />
                </ExpandableChart>
              </div>
            )
          : null
      }


      {
        result.chart_type ===
          "scatter" &&
        chartData.length >
          0
          ? (
              <div
                className={
                  styles.chartPanel
                }
              >
                <ExpandableChart
                  title={
                    result.title
                  }
                >
                  <ScatterPlot
                    data={
                      chartData
                    }
                    xLabel={
                      friendlyVariableLabel(
                        xColumn ??
                        "Variable X"
                      )
                    }
                    yLabel={
                      friendlyVariableLabel(
                        yColumn ??
                        "Variable Y"
                      )
                    }
                  />
                </ExpandableChart>
              </div>
            )
          : null
      }


      {
        result.chart_type ===
          "lorenz" &&
        chartData.length >
          0
          ? (
              <div
                className={
                  styles.chartPanel
                }
              >
                <ExpandableChart
                  title={
                    result.title
                  }
                >
                  <RequestedLorenzChart
                    data={
                      chartData
                    }
                  />
                </ExpandableChart>
              </div>
            )
          : null
      }


      {
        result.chart_type ===
          "heatmap" &&
        chartData.length >
          0
          ? (
              <div
                className={
                  styles.chartPanel
                }
              >
                <ExpandableChart
                  title={
                    result.title
                  }
                >
                  <RequestedHeatmapChart
                    data={
                      chartData
                    }
                    xLabel={
                      friendlyVariableLabel(
                        xColumn ??
                        "Variable X"
                      )
                    }
                    yLabel={
                      friendlyVariableLabel(
                        yColumn ??
                        "Variable Y"
                      )
                    }
                  />
                </ExpandableChart>
              </div>
            )
          : null
      }


      {
        result.chart_type ===
          "bar" &&
        chartData.length >
          0
          ? (
              <div
                className={
                  styles.chartPanel
                }
              >
                <div
                  className={
                    styles.chartHeader
                  }
                >
                  <div>
                    <h3>
                      {
                        result.family ===
                          "ranking"
                          ? "Classement demandé"
                          : "Agrégation demandée"
                      }
                    </h3>

                    <p>
                      {
                        result.family ===
                          "ranking"
                          ? (
                              "Agrégation, tri et limite calculés " +
                              "par le moteur Python déterministe."
                            )
                          : (
                              "Valeurs agrégées calculées par le " +
                              "moteur Python déterministe."
                            )
                      }
                    </p>
                  </div>
                </div>

                <ExpandableChart
                  title={
                    result.title
                  }
                >
                  <RequestedBarChart
                    data={
                      chartData
                    }
                    categoryLabel={
                      productColumn
                        ? "Produit"
                        : friendlyVariableLabel(
                            dimensionColumn ??
                            "Catégorie"
                          )
                    }
                    valueLabel={
                      friendlyVariableLabel(
                        valueColumn ??
                        "Valeur agrégée"
                      )
                    }
                  />
                </ExpandableChart>
              </div>
            )
          : null
      }


      {
        result.chart_type ===
          "boxplot" &&
        chartData.length >
          0
          ? (
              <div
                className={
                  styles.chartPanel
                }
              >
                <ExpandableChart
                  title={
                    result.title
                  }
                >
                  <RequestedBoxPlotChart
                    data={
                      chartData
                    }
                    groupLabel={
                      friendlyVariableLabel(
                        groupColumn ??
                        "Groupe"
                      )
                    }
                    valueLabel={
                      friendlyVariableLabel(
                        valueColumn ??
                        "Valeur"
                      )
                    }
                  />
                </ExpandableChart>
              </div>
            )
          : null
      }


      <details
        className={
          styles.technicalPanel
        }
        style={{
          marginTop:
            "12px",
        }}
      >
        <summary>
          Traçabilité de l’analyse demandée
        </summary>

        <div
          className={
            styles.evidenceFlow
          }
        >
          <article
            className={
              styles.evidenceItem
            }
          >
            <span>
              Planner
            </span>

            <strong>
              {
                report.planner_model
              }
            </strong>

            <small>
              {
                report.planner
                  .attempt_count ??
                1
              }
              {" tentative(s)"}
            </small>
          </article>


          <article
            className={
              styles.evidenceItem
            }
          >
            <span>
              Validation
            </span>

            <strong>
              Python
            </strong>

            <small>
              Contrat analytique validé
            </small>
          </article>


          <article
            className={
              styles.evidenceItem
            }
          >
            <span>
              Tool calling
            </span>

            <strong>
              {
                nativeTool
                  .requested_tool ??
                "—"
              }
            </strong>

            <small>
              {
                report.tool_model
              }
            </small>
          </article>


          <article
            className={
              styles.evidenceItem
            }
          >
            <span>
              Exécution
            </span>

            <strong>
              Python
            </strong>

            <small>
              {
                nativeTool.attempt_count
              }
              {" tentative(s) · "}
              {
                nativeTool.retry_count
              }
              {" retry"}
            </small>
          </article>
        </div>


        {
          result.family ===
            "time_series"
            ? (
                <div
                  className={
                    styles.technicalReasons
                  }
                >
                  <p>
                    {
                      aggregationFunction ===
                        "median"
                        ? (
                            <>
                              Exécution temporelle déterministe :
                              médiane de
                              {" "}
                              <strong>
                                {
                                  friendlyVariableLabel(
                                    valueColumn ??
                                    "Valeur"
                                  )
                                }
                              </strong>
                              {" "}
                              regroupée par
                              {" "}
                              <strong>
                                {
                                  friendlyVariableLabel(
                                    timeColumn ??
                                    "Temps"
                                  )
                                }
                              </strong>
                              , avec Q1 et Q3 calculés pour chaque
                              période.
                            </>
                          )
                        : (
                            <>
                              Exécution temporelle déterministe :
                              agrégation
                              {" "}
                              <strong>
                                {
                                  aggregationFunction ??
                                  "validée"
                                }
                              </strong>
                              {" "}
                              de
                              {" "}
                              <strong>
                                {
                                  friendlyVariableLabel(
                                    valueColumn ??
                                    "Valeur"
                                  )
                                }
                              </strong>
                              {" "}
                              regroupée par
                              {" "}
                              <strong>
                                {
                                  friendlyVariableLabel(
                                    timeColumn ??
                                    "Temps"
                                  )
                                }
                              </strong>
                              .
                            </>
                          )
                    }
                  </p>
                </div>
              )
            : null
        }


        {
          (
            report
              .planner
              .normalization_count ??
            0
          ) >
          0
            ? (
                <div
                  className={
                    styles.technicalReasons
                  }
                >
                  <p>
                    Python a appliqué
                    {" "}
                    {
                      report
                        .planner
                        .normalization_count
                    }
                    {" "}
                    normalisation(s) de protocole
                    avant validation.
                  </p>
                </div>
              )
            : null
        }
      </details>
    </section>
  );
}
export default function NativeRequestedAnalysisCard({
  report,
  objective,
}: {
  report:
    AINativePipelineReportView;

  objective:
    string;
}) {
  const executedItems =
    report.items.filter(
      (
        candidate
      ) =>
        candidate.pipeline_status ===
          "executed" &&
        candidate.native_tool
          ?.execution
          ?.result
          !==
          null
    );


  if (
    executedItems.length ===
    0
  ) {
    return (
      <NativeRequestedAnalysisSingleCard
        report={
          report
        }
        objective={
          objective
        }
      />
    );
  }


  return (
    <section
      className={
        styles.requestedAnswerGroup
      }
      aria-labelledby="requested-answer-group-title"
    >
      <div
        className={
          styles.requestedAnswerGroupHeader
        }
      >
        <div>
          <span
            className={
              styles.eyebrow
            }
          >
            Votre demande
          </span>

          <h2
            id="requested-answer-group-title"
          >
            Réponse à votre demande
          </h2>

          <p
            className={
              styles.requestedQuestion
            }
          >
            «
            {
              objective.trim() ||
              report.planner.objective
            }
            »
          </p>
        </div>

        <span
          className={
            styles.requestedAnswerCount
          }
        >
          {
            executedItems.length
          }
          {" résultat"}
          {
            executedItems.length >
            1
              ? "s"
              : ""
          }
          {" vérifié"}
          {
            executedItems.length >
            1
              ? "s"
              : ""
          }
        </span>
      </div>


      <div
        className={
          styles.requestedAnswerStack
        }
      >
        {
          executedItems.map(
            (
              executedItem
            ) => (
              <NativeRequestedAnalysisSingleCard
                report={{
                  ...report,
                  items: [
                    executedItem,
                  ],
                }}
                objective={
                  objective
                }
                key={
                  executedItem.contract_id
                }
              />
            )
          )
        }
      </div>
    </section>
  );
}

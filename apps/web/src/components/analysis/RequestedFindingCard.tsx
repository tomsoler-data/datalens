"use client";

import type {
  FindingRagContext,
  ReportRequestedFinding,
} from "../../app/types";

import type {
  ReportAvailableAnalysisDetailView,
  RequestedTimeGranularity,
} from "./analysisTypes";

import {
  familyLabel,
} from "./analysisPlanningPresentation";

import {
  formatChartNumber,
  formatDecimal,
  formatNumber,
  friendlyVariableLabel,
  metricNumber,
  metricString,
} from "./analysisPresentation";

import ExpandableChart from "./charts/ExpandableChart";
import ScatterPlot from "./charts/ScatterPlot";
import RequestedTimeSeriesChart from "./charts/RequestedTimeSeriesChart";
import RequestedBarChart from "./charts/RequestedBarChart";
import RequestedLorenzChart from "./charts/RequestedLorenzChart";
import RequestedHeatmapChart from "./charts/RequestedHeatmapChart";
import RequestedBoxPlotChart from "./charts/RequestedBoxPlotChart";

import styles from "../../app/page.module.css";


function requestedAggregationPeriodLabel(
  value:
    string |
    null
): string | null {
  const normalized =
    (
      value ??
      ""
    )
      .trim()
      .toLowerCase();


  switch (
    normalized
  ) {
    case "day":
      return "Jour";

    case "week":
      return "Semaine";

    case "month":
      return "Mois";

    case "quarter":
      return "Trimestre";

    case "year":
      return "Ann\u00e9e";

    default:
      return value &&
        value.trim()
        ? value.trim()
        : null;
  }
}


function requestedAnalysisLabel(
  finding:
    ReportRequestedFinding
): string {
  switch (
    finding.kind
  ) {
    case "age_total_amount_association":
      return "Âge × montant total des achats";

    case "age_frequency_association":
      return "Âge × fréquence d’achat";

    case "age_average_basket_association":
      return "Âge × panier moyen";

    case "gender_category_association":
      return "Genre × catégorie de produits";

    case "age_category_association":
      return "Âge × catégorie de produits";

    case "revenue_moving_average":
      return "Évolution du chiffre d’affaires";

    case "revenue_by_category":
      return "Chiffre d’affaires par catégorie";

    case "customers_by_period":
      return "Clients par période";

    case "transaction_count":
      return "Nombre de transactions";

    case "products_sold_count":
      return "Produits vendus";

    case "top_products":
      return "Meilleures ventes";

    case "flop_products":
      return "Produits les moins vendus";

    case "product_category_distribution":
      return "Répartition des catégories";

    case "b2b_revenue_distribution":
      return "Répartition du chiffre d’affaires BtoB";

    case "lorenz_curve":
      return "Concentration du chiffre d’affaires";

    default:
      return finding.title;
  }
}


function analyticalGrainLabel(
  value:
    string
): string {
  switch (
    value
  ) {
    case "customer":
      return "client";

    case "transaction":
      return "transaction";

    case "product":
      return "produit";

    case "session":
      return "session";

    default:
      return friendlyVariableLabel(
        value
      );
  }
}


function descriptiveAssociationMagnitude(
  value:
    number
): string {
  const absolute =
    Math.abs(
      value
    );


  if (
    absolute <
    0.10
  ) {
    return "très faible";
  }


  if (
    absolute <
    0.30
  ) {
    return "faible";
  }


  if (
    absolute <
    0.50
  ) {
    return "modérée";
  }


  if (
    absolute <
    0.70
  ) {
    return "forte";
  }


  return "très forte";
}


function descriptiveAssociationReading(
  coefficient:
    number,

  xLabel:
    string,

  yLabel:
    string
): string {
  const magnitude =
    descriptiveAssociationMagnitude(
      coefficient
    );


  if (
    Math.abs(
      coefficient
    ) <
    0.10
  ) {
    return (
      `Dans les observations, aucune tendance monotone marquée ` +
      `n’apparaît entre ${xLabel.toLowerCase()} et ` +
      `${yLabel.toLowerCase()} (ρ = ${formatDecimal(
        coefficient
      )}).`
    );
  }


  const direction =
    coefficient >
    0
      ? "augmenter"
      : "diminuer";


  return (
    `Dans les observations, lorsque ${xLabel.toLowerCase()} augmente, ` +
    `${yLabel.toLowerCase()} tend à ${direction}. ` +
    `L’association monotone observée est ${magnitude} ` +
    `(ρ = ${formatDecimal(
      coefficient
    )}).`
  );
}


function requestedExecutionLabel(
  finding:
    ReportRequestedFinding
): string {
  switch (
    finding.execution_status
  ) {
    case "complete":
      return "Analyse complète";

    case "descriptive_only":
      return "Analyse descriptive";

    case "needs_information":
      return "Informations nécessaires";

    case "needs_specialized_method":
      return "Méthode spécialisée requise";

    default:
      return finding.execution_status
        .replace(
          /_/g,
          " "
        );
  }
}


function RequestedRagContextBlock({
  context,
}: {
  context:
    FindingRagContext |
    null;
}) {
  if (
    !context
  ) {
    return null;
  }


  if (
    context.explanation.status ===
      "ready" &&
    context.explanation.claims.length >
      0
  ) {
    return (
      <details
        className={
          styles.technicalPanel
        }
      >
        <summary>
          Éclairage métier
          {" · "}
          vérifié
        </summary>


        <div
          className={
            styles.technicalReasons
          }
        >
          <p>
            {
              context
                .explanation
                .explanation
            }
          </p>
        </div>


        <div
          className={
            styles.evidenceFlow
          }
        >
          {
            context
              .explanation
              .claims
              .map(
                (
                  claim,
                  index
                ) => (
                  <article
                    className={
                      styles.evidenceItem
                    }
                    key={
                      `${
                        claim
                          .citation
                          .chunk_id
                      }-${index}`
                    }
                  >
                    <span>
                      Preuve complémentaire
                    </span>

                    <strong>
                      {
                        claim
                          .citation
                          .filename
                      }
                    </strong>

                    <small>
                      {
                        claim
                          .evidence_quote
                      }
                    </small>

                    <small>
                      {
                        claim
                          .citation
                          .source_locator
                      }
                    </small>
                  </article>
                )
              )
          }
        </div>
      </details>
    );
  }


  return (
    <details
      className={
        styles.technicalPanel
      }
    >
      <summary>
        Éclairage métier
        {" · "}
        aucun contexte supplémentaire
      </summary>

      <div
        className={
          styles.technicalReasons
        }
      >
        <p>
          Aucun contexte documentaire
          supplémentaire suffisamment direct
          n’a été retenu pour cette analyse.
        </p>

        <p>
          La demande source reste néanmoins
          vérifiée et traçable indépendamment
          du RAG.
        </p>
      </div>
    </details>
  );
}


export default function RequestedFindingCard({
  finding,
  index,
  ragContext,
  reconfigurationAnalysis = null,
  reconfigurationLoading = false,
  reconfigurationError = null,
  onReconfigureTimeSeries,
}: {
  finding:
    ReportRequestedFinding;

  index:
    number;

  ragContext:
    FindingRagContext |
    null;

  reconfigurationAnalysis?:
    ReportAvailableAnalysisDetailView |
    null;

  reconfigurationLoading?:
    boolean;

  reconfigurationError?:
    string |
    null;

  onReconfigureTimeSeries?: (
    analysis:
      ReportAvailableAnalysisDetailView,

    timeGranularity:
      RequestedTimeGranularity,

    movingAverageWindow:
      number
  ) => Promise<
    void
  >;
}) {
  const pearson =
    metricNumber(
      finding.metrics,
      "pearson_r"
    );

  const spearman =
    metricNumber(
      finding.metrics,
      "spearman_rho"
    );

  const xColumn =
    metricString(
      finding.metrics,
      "x_column"
    );

  const yColumn =
    metricString(
      finding.metrics,
      "y_column"
    );


  const xLabel =
    xColumn
      ? friendlyVariableLabel(
          xColumn
        )
      : "Variable X";


  const yLabel =
    yColumn
      ? friendlyVariableLabel(
          yColumn
        )
      : "Variable Y";


  const descriptiveReading =
    spearman !==
      null
      ? descriptiveAssociationReading(
          spearman,
          xLabel,
          yLabel
        )
      : null;


  const requestedXColumn =
    finding.variables.x ??
    xColumn ??
    null;


  const requestedYColumn =
    finding.variables.y ??
    yColumn ??
    null;


  const requestedGroupColumn =
    finding.variables.group ??
    null;


  const requestedValueColumn =
    finding.variables.value ??
    null;


  const requestedXLabel =
    requestedXColumn
      ? friendlyVariableLabel(
          requestedXColumn
        )
      : "Variable X";


  const requestedYLabel =
    requestedYColumn
      ? friendlyVariableLabel(
          requestedYColumn
        )
      : "Variable Y";


  const requestedGroupLabel =
    requestedGroupColumn
      ? friendlyVariableLabel(
          requestedGroupColumn
        )
      : "Groupe";


  const requestedValueLabel =
    requestedValueColumn
      ? friendlyVariableLabel(
          requestedValueColumn
        )
      : "Valeur";


  const transactionCount =
    metricNumber(
      finding.metrics,
      "transaction_count"
    );


  const productsSoldCount =
    metricNumber(
      finding.metrics,
      "products_sold_count"
    );


  const distinctProductsSold =
    metricNumber(
      finding.metrics,
      "distinct_products_sold"
    );


  const totalRevenue =
    metricNumber(
      finding.metrics,
      "total_revenue"
    );


  const distinctCustomersTotal =
    metricNumber(
      finding.metrics,
      "distinct_customers_total"
    );


  const periodCount =
    metricNumber(
      finding.metrics,
      "period_count"
    );


  const categoryCount =
    metricNumber(
      finding.metrics,
      "category_count"
    );


  const movingAverageWindow =
    metricNumber(
      finding.metrics,
      "moving_average_window"
    );


  const aggregationPeriod =
    metricString(
      finding.metrics,
      "aggregation_period"
    );


  const requestedTimeGranularities:
    RequestedTimeGranularity[] =
      [
        "day",
        "week",
        "month",
        "quarter",
        "year",
      ];


  const currentTimeGranularity:
    RequestedTimeGranularity |
    null =
      aggregationPeriod !==
        null &&
      requestedTimeGranularities.includes(
        aggregationPeriod as
          RequestedTimeGranularity
      )
        ? (
            aggregationPeriod as
              RequestedTimeGranularity
          )
        : null;


  const currentMovingAverageWindow =
    movingAverageWindow !==
      null &&
    Number.isInteger(
      movingAverageWindow
    ) &&
    movingAverageWindow >=
      1
      ? movingAverageWindow
      : null;


  const canReconfigureTimeSeries =
    finding.kind ===
      "revenue_moving_average" &&
    finding.chart_type ===
      "line" &&
    finding.chart_data.length >
      0 &&
    reconfigurationAnalysis !==
      null &&
    currentTimeGranularity !==
      null &&
    currentMovingAverageWindow !==
      null &&
    typeof onReconfigureTimeSeries ===
      "function";


  const aggregationPeriodLabel =
    requestedAggregationPeriodLabel(
      aggregationPeriod
    );


  const giniCoefficient =
    metricNumber(
      finding.metrics,
      "gini_coefficient"
    );


  const referenceCount =
    metricNumber(
      finding.metrics,
      "reference_count"
    );


  const rankingLimit =
    metricNumber(
      finding.metrics,
      "ranking_limit"
    );


  const rankedProductCount =
    metricNumber(
      finding.metrics,
      "ranked_product_count"
    );


  const customerCount =
    metricNumber(
      finding.metrics,
      "customer_count"
    );


  const requestedTimeSeriesValueLabel =
    finding.kind ===
      "revenue_moving_average"
      ? "Chiffre d’affaires"
      : (
          finding.kind ===
            "customers_by_period"
            ? "Clients actifs distincts"
            : requestedValueLabel
        );


  const requestedRankingMetric =
    metricString(
      finding.metrics,
      "ranking_metric"
    );


  const requestedBarValueLabel =
    finding.kind ===
      "top_products" ||
    finding.kind ===
      "flop_products"
      ? (
          requestedRankingMetric ===
            "transaction_count"
            ? "Nombre de transactions"
            : "Chiffre d\u2019affaires"
        )
      : (
          finding.kind ===
            "revenue_by_category"
            ? "Chiffre d\u2019affaires"
            : (
                finding.kind ===
                  "product_category_distribution"
                  ? "R\u00e9f\u00e9rences distinctes"
                  : requestedValueLabel
              )
        );


  const requestedBarCategoryLabel =
    (
      finding.kind ===
        "top_products" ||
      finding.kind ===
        "flop_products"
    )
      ? "Produit"
      : (
          finding.kind ===
            "revenue_by_category" ||
          finding.kind ===
            "product_category_distribution"
            ? "Catégorie"
            : requestedGroupLabel
        );


  const requestedBarTitle =
    finding.kind ===
      "top_products"
      ? "Top produits"
      : (
          finding.kind ===
            "flop_products"
            ? "Flop produits"
            : (
                finding.kind ===
                  "product_category_distribution"
                  ? "Références par catégorie"
                  : "Chiffre d’affaires par catégorie"
              )
        );


  const requestedBarDescription =
    finding.kind ===
      "top_products"
      ? (
          requestedRankingMetric ===
            "transaction_count"
            ? "R\u00e9f\u00e9rences class\u00e9es par nombre de transactions distinctes d\u00e9croissant."
            : "R\u00e9f\u00e9rences class\u00e9es par chiffre d\u2019affaires d\u00e9croissant."
        )
      : (
          finding.kind ===
            "flop_products"
            ? (
                requestedRankingMetric ===
                  "transaction_count"
                  ? "R\u00e9f\u00e9rences class\u00e9es par nombre de transactions distinctes croissant."
                  : "R\u00e9f\u00e9rences class\u00e9es par chiffre d\u2019affaires croissant."
              )
            : (
                finding.kind ===
                  "product_category_distribution"
                  ? "Nombre de r\u00e9f\u00e9rences distinctes observ\u00e9es dans chaque cat\u00e9gorie."
                  : "Comparaison descriptive du chiffre d\u2019affaires agr\u00e9g\u00e9 par cat\u00e9gorie."
              )
        );


  return (
    <article
      className={
        styles.explanationCard
      }
      style={{
        gridColumn:
          "1 / -1",
      }}
    >
      <div
        className={
          styles.chartHeader
        }
      >
        <div>
          <span
            className={
              styles.eyebrow
            }
          >
            Demande
            {" "}
            {
              String(
                index + 1
              ).padStart(
                2,
                "0"
              )
            }
            {" · "}
            {
              requestedExecutionLabel(
                finding
              )
            }
          </span>


          <h3
            className={
              styles.explanationTitle
            }
          >
            {
              requestedAnalysisLabel(
                finding
              )
            }
          </h3>


          <p
            className={
              styles.resultSubtitle
            }
          >
            {
              familyLabel(
                finding.family
              )
            }

            {
              finding.analytical_grain
                ? (
                    <>
                      {" · "}
                      grain
                      {" "}
                      {
                        analyticalGrainLabel(
                          finding.analytical_grain
                        )
                      }
                    </>
                  )
                : null
            }
          </p>


          {
            xColumn &&
            yColumn
              ? (
                  <p
                    className={
                      styles.resultSubtitle
                    }
                  >
                    {
                      friendlyVariableLabel(
                        xColumn
                      )
                    }
                    {" × "}
                    {
                      friendlyVariableLabel(
                        yColumn
                      )
                    }
                  </p>
                )
              : null
          }
        </div>
      </div>


      <div
        className={
          styles.metricGrid
        }
      >
        {
          spearman !==
            null
            ? (
                <article
                  className={
                    styles.metricCard
                  }
                >
                  <span>
                    Spearman ρ
                  </span>

                  <strong>
                    {
                      formatDecimal(
                        spearman
                      )
                    }
                  </strong>
                </article>
              )
            : null
        }


        {
          pearson !==
            null
            ? (
                <article
                  className={
                    styles.metricCard
                  }
                >
                  <span>
                    Pearson r
                  </span>

                  <strong>
                    {
                      formatDecimal(
                        pearson
                      )
                    }
                  </strong>
                </article>
              )
            : null
        }


        <article
          className={
            styles.metricCard
          }
        >
          <span>
            Observations
          </span>

          <strong>
            {
              formatNumber(
                finding.sample_size
              )
            }
          </strong>
        </article>


        <article
          className={
            styles.metricCard
          }
        >
          <span>
            Inférence
          </span>

          <strong
            className={
              finding.inferential_status ===
                "executed"
                ? styles.statusGood
                : styles.statusNeutral
            }
          >
            {
              finding.inferential_status ===
                "executed"
                ? "Exécutée"
                : (
                    finding.inferential_status ===
                      "not_applicable"
                      ? "Non applicable"
                      : "Non sélectionnée"
                  )
            }
          </strong>
        </article>
      </div>


      {
        (
          transactionCount !==
            null ||
          productsSoldCount !==
            null ||
          distinctProductsSold !==
            null ||
          totalRevenue !==
            null ||
          distinctCustomersTotal !==
            null ||
          periodCount !==
            null ||
          categoryCount !==
            null ||
          (
            finding.kind !==
              "revenue_moving_average" &&
            aggregationPeriodLabel !==
              null
          ) ||
          (
            finding.kind !==
              "revenue_moving_average" &&
            movingAverageWindow !==
              null
          ) ||
          giniCoefficient !==
            null ||
          referenceCount !==
            null ||
          rankingLimit !==
            null ||
          rankedProductCount !==
            null ||
          customerCount !==
            null
        )
          ? (
              <div
                className={
                  styles.metricGrid
                }
                style={{
                  marginTop:
                    "12px",
                }}
              >
                {
                  totalRevenue !==
                    null
                    ? (
                        <article
                          className={
                            styles.metricCard
                          }
                        >
                          <span>
                            Chiffre d’affaires
                          </span>

                          <strong>
                            {
                              formatChartNumber(
                                totalRevenue
                              )
                            }
                          </strong>
                        </article>
                      )
                    : null
                }


                {
                  transactionCount !==
                    null
                    ? (
                        <article
                          className={
                            styles.metricCard
                          }
                        >
                          <span>
                            Transactions
                          </span>

                          <strong>
                            {
                              formatNumber(
                                transactionCount
                              )
                            }
                          </strong>
                        </article>
                      )
                    : null
                }


                {
                  productsSoldCount !==
                    null
                    ? (
                        <article
                          className={
                            styles.metricCard
                          }
                        >
                          <span>
                            Produits vendus
                          </span>

                          <strong>
                            {
                              formatNumber(
                                productsSoldCount
                              )
                            }
                          </strong>
                        </article>
                      )
                    : null
                }


                {
                  distinctProductsSold !==
                    null
                    ? (
                        <article
                          className={
                            styles.metricCard
                          }
                        >
                          <span>
                            Références distinctes
                          </span>

                          <strong>
                            {
                              formatNumber(
                                distinctProductsSold
                              )
                            }
                          </strong>
                        </article>
                      )
                    : null
                }


                {
                  distinctCustomersTotal !==
                    null
                    ? (
                        <article
                          className={
                            styles.metricCard
                          }
                        >
                          <span>
                            Clients uniques sur toute la période
                          </span>

                          <strong>
                            {
                              formatNumber(
                                distinctCustomersTotal
                              )
                            }
                          </strong>
                        </article>
                      )
                    : null
                }


                {
                  periodCount !==
                    null
                    ? (
                        <article
                          className={
                            styles.metricCard
                          }
                        >
                          <span>
                            Périodes
                          </span>

                          <strong>
                            {
                              formatNumber(
                                periodCount
                              )
                            }
                          </strong>
                        </article>
                      )
                    : null
                }


                {
                  categoryCount !==
                    null
                    ? (
                        <article
                          className={
                            styles.metricCard
                          }
                        >
                          <span>
                            Catégories
                          </span>

                          <strong>
                            {
                              formatNumber(
                                categoryCount
                              )
                            }
                          </strong>
                        </article>
                      )
                    : null
                }


                {
                  finding.kind !==
                    "revenue_moving_average" &&
                  aggregationPeriodLabel !==
                    null
                    ? (
                        <article
                          className={
                            styles.metricCard
                          }
                        >
                          <span>
                            {
                              "P\u00e9riode d\u2019agr\u00e9gation"
                            }
                          </span>

                          <strong>
                            {
                              aggregationPeriodLabel
                            }
                          </strong>
                        </article>
                      )
                    : null
                }


                {
                  finding.kind !==
                    "revenue_moving_average" &&
                  movingAverageWindow !==
                    null
                    ? (
                        <article
                          className={
                            styles.metricCard
                          }
                        >
                          <span>
                            Fenêtre moyenne mobile
                          </span>

                          <strong>
                            {
                              `${formatNumber(
                                movingAverageWindow
                              )} périodes`
                            }
                          </strong>
                        </article>
                      )
                    : null
                }


                {
                  giniCoefficient !==
                    null
                    ? (
                        <article
                          className={
                            styles.metricCard
                          }
                        >
                          <span>
                            Coefficient de Gini
                          </span>

                          <strong>
                            {
                              formatDecimal(
                                giniCoefficient
                              )
                            }
                          </strong>
                        </article>
                      )
                    : null
                }


                {
                  referenceCount !==
                    null
                    ? (
                        <article
                          className={
                            styles.metricCard
                          }
                        >
                          <span>
                            Références distinctes
                          </span>

                          <strong>
                            {
                              formatNumber(
                                referenceCount
                              )
                            }
                          </strong>
                        </article>
                      )
                    : null
                }


                {
                  rankingLimit !==
                    null
                    ? (
                        <article
                          className={
                            styles.metricCard
                          }
                        >
                          <span>
                            Références affichées
                          </span>

                          <strong>
                            {
                              formatNumber(
                                rankingLimit
                              )
                            }
                          </strong>
                        </article>
                      )
                    : null
                }


                {
                  rankedProductCount !==
                    null
                    ? (
                        <article
                          className={
                            styles.metricCard
                          }
                        >
                          <span>
                            Références classées
                          </span>

                          <strong>
                            {
                              formatNumber(
                                rankedProductCount
                              )
                            }
                          </strong>
                        </article>
                      )
                    : null
                }


                {
                  customerCount !==
                    null
                    ? (
                        <article
                          className={
                            styles.metricCard
                          }
                        >
                          <span>
                            Clients dans Lorenz
                          </span>

                          <strong>
                            {
                              formatNumber(
                                customerCount
                              )
                            }
                          </strong>
                        </article>
                      )
                    : null
                }
              </div>
            )
          : null
      }


      {
        descriptiveReading
          ? (
              <section
                className={
                  styles.summaryPanel
                }
                style={{
                  marginTop:
                    "12px",
                }}
              >
                <div
                  className={
                    styles.summaryItem
                  }
                >
                  <span>
                    Lecture descriptive
                  </span>

                  <p>
                    {
                      descriptiveReading
                    }
                  </p>

                  <small>
                    Repère générique fondé sur
                    la valeur absolue de Spearman ρ.
                    Cette lecture ne constitue ni
                    une preuve de causalité ni une
                    conclusion de significativité
                    statistique.
                  </small>
                </div>
              </section>
            )
          : null
      }


      {
        finding.summary.length >
        0
          ? (
              <details
                className={
                  styles.technicalPanel
                }
              >
                <summary>
                  Détail du résultat
                </summary>

                <div
                  className={
                    styles.technicalReasons
                  }
                >
                  {
                    finding.summary.map(
                      (
                        item
                      ) => (
                        <p
                          key={
                            item
                          }
                        >
                          {
                            item
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


      {
        finding.chart_type ===
          "line" &&
        finding.chart_data.length >
          0
          ? (
              <div
                className={
                  styles.chartPanel
                }
                style={{
                  marginTop:
                    "12px",
                }}
              >
                <div
                  className={
                    styles.chartHeader
                  }
                >
                  <h3>
                    Évolution temporelle
                  </h3>

                  <p>
                    {
                      finding.kind ===
                        "revenue_moving_average"
                        ? (
                            "La ligne continue montre la valeur par période ; " +
                            "la ligne pointillée montre la moyenne mobile."
                          )
                        : (
                            "\u00c9volution du nombre de clients actifs distincts par mois."
                          )
                    }
                  </p>
                </div>

                {
                  finding.kind ===
                    "revenue_moving_average" &&
                  (
                    aggregationPeriodLabel !==
                      null ||
                    currentMovingAverageWindow !==
                      null
                  )
                    ? (
                        <div
                          style={{
                            display:
                              "grid",

                            gridTemplateColumns:
                              "repeat(auto-fit, minmax(180px, 1fr))",

                            gap:
                              "10px",

                            margin:
                              "12px 0 14px",
                          }}
                        >
                          <article
                            style={{
                              display:
                                "grid",

                              gap:
                                "5px",

                              padding:
                                "12px 14px",

                              border:
                                "1px solid rgba(116, 177, 255, 0.18)",

                              borderRadius:
                                "10px",

                              background:
                                "rgba(35, 78, 128, 0.055)",
                            }}
                          >
                            <span
                              style={{
                                color:
                                  "#92a8c5",

                                fontSize:
                                  "0.66rem",

                                fontWeight:
                                  600,

                                textTransform:
                                  "uppercase",

                                letterSpacing:
                                  "0.06em",
                              }}
                            >
                              P?riode
                            </span>

                            <strong
                              style={{
                                color:
                                  "#dce6f3",

                                fontSize:
                                  "0.92rem",

                                fontWeight:
                                  700,
                              }}
                            >
                              {
                                aggregationPeriodLabel ??
                                "Non disponible"
                              }
                            </strong>
                          </article>


                          <article
                            style={{
                              display:
                                "grid",

                              gap:
                                "5px",

                              padding:
                                "12px 14px",

                              border:
                                "1px solid rgba(232, 184, 97, 0.36)",

                              borderRadius:
                                "10px",

                              background:
                                "linear-gradient(135deg, rgba(232, 184, 97, 0.10), rgba(232, 184, 97, 0.035))",

                              boxShadow:
                                "0 8px 24px rgba(232, 184, 97, 0.055)",
                            }}
                          >
                            <span
                              style={{
                                color:
                                  "#d8b66f",

                                fontSize:
                                  "0.66rem",

                                fontWeight:
                                  650,

                                textTransform:
                                  "uppercase",

                                letterSpacing:
                                  "0.06em",
                              }}
                            >
                              Moyenne mobile
                            </span>

                            <strong
                              style={{
                                color:
                                  "#f0c979",

                                fontSize:
                                  "0.92rem",

                                fontWeight:
                                  750,
                              }}
                            >
                              {
                                currentMovingAverageWindow !==
                                  null
                                  ? (
                                      `${formatNumber(
                                        currentMovingAverageWindow
                                      )} p?riodes`
                                    )
                                  : (
                                      "Non disponible"
                                    )
                              }
                            </strong>
                          </article>
                        </div>
                      )
                    : null
                }


                {
                  canReconfigureTimeSeries &&
                  reconfigurationAnalysis &&
                  currentTimeGranularity !==
                    null &&
                  currentMovingAverageWindow !==
                    null &&
                  onReconfigureTimeSeries
                    ? (
                        <form
                          key={
                            `${finding.analysis_id}-${currentTimeGranularity}-${currentMovingAverageWindow}`
                          }
                          onSubmit={
                            (
                              event
                            ) => {
                              event.preventDefault();


                              const formData =
                                new FormData(
                                  event.currentTarget
                                );


                              const rawGranularity =
                                String(
                                  formData.get(
                                    "reconfigure_time_granularity"
                                  ) ??
                                  ""
                                );


                              if (
                                !requestedTimeGranularities.includes(
                                  rawGranularity as
                                    RequestedTimeGranularity
                                )
                              ) {
                                return;
                              }


                              const windowValue =
                                Number(
                                  formData.get(
                                    "reconfigure_moving_average_window"
                                  )
                                );


                              if (
                                !Number.isInteger(
                                  windowValue
                                ) ||
                                windowValue <
                                  1
                              ) {
                                return;
                              }


                              void onReconfigureTimeSeries(
                                reconfigurationAnalysis,
                                rawGranularity as
                                  RequestedTimeGranularity,
                                windowValue
                              );
                            }
                          }
                          style={{
                            margin:
                              "12px 0 14px",

                            padding:
                              "12px",

                            border:
                              "1px solid rgba(116, 177, 255, 0.16)",

                            borderRadius:
                              "10px",

                            background:
                              "rgba(35, 78, 128, 0.055)",
                          }}
                        >
                          <strong
                            style={{
                              display:
                                "block",

                              color:
                                "#dce6f3",

                              fontSize:
                                "0.72rem",

                              fontWeight:
                                650,
                            }}
                          >
                            Modifier les paramètres temporels
                          </strong>


                          <p
                            style={{
                              margin:
                                "4px 0 0",

                              color:
                                "#92a8c5",

                              fontSize:
                                "0.67rem",

                              lineHeight:
                                1.5,
                            }}
                          >
                            DataLens recalcule cette analyse depuis
                            l&apos;artefact validé côté serveur.
                          </p>


                          <div
                            style={{
                              display:
                                "grid",

                              gridTemplateColumns:
                                "repeat(auto-fit, minmax(145px, 1fr))",

                              gap:
                                "10px",

                              marginTop:
                                "11px",

                              alignItems:
                                "end",
                            }}
                          >
                            <label
                              style={{
                                display:
                                  "grid",

                                gap:
                                  "5px",

                                color:
                                  "#aebdd0",

                                fontSize:
                                  "0.66rem",
                              }}
                            >
                              Période

                              <select
                                name="reconfigure_time_granularity"
                                defaultValue={
                                  currentTimeGranularity
                                }
                                disabled={
                                  reconfigurationLoading
                                }
                              >
                                <option value="day">
                                  Jour
                                </option>

                                <option value="week">
                                  Semaine
                                </option>

                                <option value="month">
                                  Mois
                                </option>

                                <option value="quarter">
                                  Trimestre
                                </option>

                                <option value="year">
                                  Année
                                </option>
                              </select>
                            </label>


                            <label
                              style={{
                                display:
                                  "grid",

                                gap:
                                  "5px",

                                color:
                                  "#aebdd0",

                                fontSize:
                                  "0.66rem",
                              }}
                            >
                              Fenêtre mobile

                              <input
                                name="reconfigure_moving_average_window"
                                type="number"
                                min={
                                  1
                                }
                                step={
                                  1
                                }
                                defaultValue={
                                  currentMovingAverageWindow
                                }
                                required
                                disabled={
                                  reconfigurationLoading
                                }
                              />
                            </label>


                            <button
                              type="submit"
                              disabled={
                                reconfigurationLoading
                              }
                            >
                              {
                                reconfigurationLoading
                                  ? (
                                      "Recalcul en cours..."
                                    )
                                  : (
                                      "Mettre à jour"
                                    )
                              }
                            </button>
                          </div>


                          {
                            reconfigurationError
                              ? (
                                  <p
                                    style={{
                                      margin:
                                        "9px 0 0",

                                      color:
                                        "#efaaaa",

                                      fontSize:
                                        "0.66rem",
                                    }}
                                  >
                                    {
                                      reconfigurationError
                                    }
                                  </p>
                                )
                              : null
                          }
                        </form>
                      )
                    : null
                }


                <ExpandableChart
                  title={
                    requestedAnalysisLabel(
                      finding
                    )
                  }
                >
                  <RequestedTimeSeriesChart
                    data={
                      finding.chart_data
                    }
                    valueLabel={
                      requestedTimeSeriesValueLabel
                    }
                    showMovingAverage={
                      finding.kind ===
                        "revenue_moving_average"
                    }
                  />
                </ExpandableChart>
              </div>
            )
          : null
      }


      {
        finding.chart_type ===
          "bar" &&
        finding.chart_data.length >
          0
          ? (
              <div
                className={
                  styles.chartPanel
                }
                style={{
                  marginTop:
                    "12px",
                }}
              >
                <div
                  className={
                    styles.chartHeader
                  }
                >
                  <h3>
                    {
                      requestedBarTitle
                    }
                  </h3>

                  <p>
                    {
                      requestedBarDescription
                    }
                  </p>
                </div>

                <ExpandableChart
                  title={
                    requestedBarTitle
                  }
                >
                  <RequestedBarChart
                    data={
                      finding.chart_data
                    }
                    categoryLabel={
                      requestedBarCategoryLabel
                    }
                    valueLabel={
                      requestedBarValueLabel
                    }
                  />
                </ExpandableChart>
              </div>
            )
          : null
      }


      {
        finding.chart_type ===
          "scatter" &&
        finding.chart_data.length >
          0
          ? (
              <div
                className={
                  styles.chartPanel
                }
                style={{
                  marginTop:
                    "12px",
                }}
              >
                <div
                  className={
                    styles.chartHeader
                  }
                >
                  <h3>
                    Relation observée
                  </h3>

                  <p>
                    Visualisation descriptive
                    des observations utilisées.
                  </p>
                </div>

                <ExpandableChart
                  title={
                    requestedAnalysisLabel(
                      finding
                    )
                  }
                >
                  <ScatterPlot
                    data={
                      finding.chart_data
                    }
                    xLabel={
                      xLabel
                    }
                    yLabel={
                      yLabel
                    }
                  />
                </ExpandableChart>
              </div>
            )
          : null
      }


      {
        finding.chart_type ===
          "lorenz" &&
        finding.chart_data.length >
          0
          ? (
              <div
                className={
                  styles.chartPanel
                }
                style={{
                  marginTop:
                    "12px",
                }}
              >
                <div
                  className={
                    styles.chartHeader
                  }
                >
                  <h3>
                    Courbe de Lorenz
                  </h3>

                  <p>
                    La ligne pointillée représente
                    l’égalité parfaite. Plus la
                    courbe observée s’en éloigne,
                    plus le chiffre d’affaires est
                    concentré entre certains clients.
                  </p>
                </div>

                <ExpandableChart
                  title="Courbe de Lorenz"
                >
                  <RequestedLorenzChart
                    data={
                      finding.chart_data
                    }
                  />
                </ExpandableChart>
              </div>
            )
          : null
      }


      {
        finding.chart_type ===
          "heatmap" &&
        finding.chart_data.length >
          0
          ? (
              <div
                className={
                  styles.chartPanel
                }
                style={{
                  marginTop:
                    "12px",
                }}
              >
                <div
                  className={
                    styles.chartHeader
                  }
                >
                  <h3>
                    Table de contingence
                  </h3>

                  <p>
                    Plus une cellule est intense,
                    plus le nombre d’observations
                    est élevé.
                  </p>
                </div>

                <ExpandableChart
                  title="Table de contingence"
                >
                  <RequestedHeatmapChart
                    data={
                      finding.chart_data
                    }
                    xLabel={
                      requestedXLabel
                    }
                    yLabel={
                      requestedYLabel
                    }
                  />
                </ExpandableChart>
              </div>
            )
          : null
      }


      {
        finding.chart_type ===
          "boxplot" &&
        finding.chart_data.length >
          0
          ? (
              <div
                className={
                  styles.chartPanel
                }
                style={{
                  marginTop:
                    "12px",
                }}
              >
                <div
                  className={
                    styles.chartHeader
                  }
                >
                  <h3>
                    Distribution par groupe
                  </h3>

                  <p>
                    Min, quartiles, médiane et max
                    sont affichés pour chaque groupe.
                  </p>
                </div>

                <ExpandableChart
                  title="Distribution par groupe"
                >
                  <RequestedBoxPlotChart
                    data={
                      finding.chart_data
                    }
                    groupLabel={
                      requestedGroupLabel
                    }
                    valueLabel={
                      requestedValueLabel
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
      >
        <summary>
          Demande source
          {" · "}
          vérifiée
        </summary>


        <div
          className={
            styles.technicalReasons
          }
        >
          <p>
            «
            {
              finding.evidence_quote
                .replace(
                  /,\s*$/,
                  ""
                )
            }
            »
          </p>
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
              Document
            </span>

            <strong>
              {
                finding.source_filename
              }
            </strong>

            <small>
              {
                finding.source_locator
              }
            </small>
          </article>


          <article
            className={
              styles.evidenceItem
            }
          >
            <span>
              Page
            </span>

            <strong>
              {
                finding.page_number ??
                "—"
              }
            </strong>

            <small>
              Localisation vérifiée
            </small>
          </article>


          <article
            className={
              styles.evidenceItem
            }
          >
            <span>
              Unité de preuve
            </span>

            <strong>
              {
                finding.evidence_unit_id ??
                "—"
              }
            </strong>

            <small>
              Élément documentaire
              ayant déclenché l’analyse
            </small>
          </article>


          <article
            className={
              styles.evidenceItem
            }
          >
            <span>
              Provenance
            </span>

            <strong>
              Vérifiée
            </strong>

            <small>
              {
                finding.adapter_rule_version
              }
            </small>
          </article>
        </div>
      </details>


      <RequestedRagContextBlock
        context={
          ragContext
        }
      />


      {
        finding.caveats.length >
        0
          ? (
              <details
                className={
                  styles.technicalPanel
                }
              >
                <summary>
                  Limites et précautions
                </summary>

                <div
                  className={
                    styles.technicalReasons
                  }
                >
                  {
                    finding.caveats.map(
                      (
                        caveat
                      ) => (
                        <p
                          key={
                            caveat
                          }
                        >
                          {
                            caveat
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
    </article>
  );
}

import type { UnifiedAnalysisReport }
  from "../../app/types";

import {
  datumLabel,
  datumNumber,
  formatDecimal,
  friendlyVariableLabel,
  metricNumber,
  metricString,
} from "./analysisPresentation";


type SignalKpi = {
  key: string;
  label: string;
  value: string;
  context: string;
};


export function buildSignalKpis(
  report:
    UnifiedAnalysisReport
): SignalKpi[] {
  const output:
    SignalKpi[] = [];


  for (
    const finding
    of report.main_findings
  ) {
    if (
      finding.family ===
      "time_series"
    ) {
      const change =
        metricNumber(
          finding.metrics,
          "median_change"
        );

      const measure =
        metricString(
          finding.metrics,
          "measure_column"
        );


      if (
        change !==
        null
      ) {
        output.push(
          {
            key:
              `${
                finding.analysis_id ??
                finding.title
              }-change`,

            label:
              "Évolution médiane",

            value:
              formatDecimal(
                change
              ),

            context:
              measure
                ? friendlyVariableLabel(
                    measure
                  )
                : finding.title,
          }
        );
      }
    }


    if (
      finding.family ===
      "derived_gap"
    ) {
      const gap =
        metricNumber(
          finding.metrics,
          "median_gap"
        );


      if (
        gap !==
        null
      ) {
        output.push(
          {
            key:
              `${
                finding.analysis_id ??
                finding.title
              }-gap`,

            label:
              "Écart médian",

            value:
              formatDecimal(
                gap
              ),

            context:
              finding.title,
          }
        );
      }
    }


    if (
      finding.family ===
      "quantitative_association"
    ) {
      const association =
        metricNumber(
          finding.metrics,
          "median_period_spearman"
        ) ??
        metricNumber(
          finding.metrics,
          "overall_preliminary_spearman"
        );


      if (
        association !==
        null
      ) {
        output.push(
          {
            key:
              `${
                finding.analysis_id ??
                finding.title
              }-association`,

            label:
              "Association médiane",

            value:
              formatDecimal(
                association
              ),

            context:
              finding.title,
          }
        );
      }
    }


    if (
      finding.family ===
      "group_comparison"
    ) {
      const rows =
        (
          finding.chart_data ??
          []
        )
          .map(
            (
              datum
            ) => {
              const group =
                datumLabel(
                  datum,
                  "group"
                );

              const median =
                datumNumber(
                  datum,
                  "median"
                );


              if (
                !group ||
                median ===
                  null
              ) {
                return null;
              }


              return {
                group,
                median,
              };
            }
          )
          .filter(
            (
              item
            ): item is {
              group:
                string;

              median:
                number;
            } =>
              item !==
              null
          );


      if (
        rows.length >
        0
      ) {
        const lowest =
          [...rows]
            .sort(
              (
                left,
                right
              ) =>
                left.median -
                right.median
            )[
              0
            ];


        const measure =
          metricString(
            finding.metrics,
            "measure_column"
          );


        output.push(
          {
            key:
              `${
                finding.analysis_id ??
                finding.title
              }-group`,

            label:
              `Médiane · ${lowest.group}`,

            value:
              formatDecimal(
                lowest.median
              ),

            context:
              measure
                ? friendlyVariableLabel(
                    measure
                  )
                : finding.title,
          }
        );
      }
    }


    if (
      output.length >=
      4
    ) {
      break;
    }
  }


  return output;
}

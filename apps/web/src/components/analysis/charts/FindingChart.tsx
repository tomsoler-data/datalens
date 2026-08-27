import type {
  ReportFinding,
} from "../../../app/types";

import styles from "../../../app/page.module.css";

import {
  friendlyVariableLabel,
  metricString,
} from "../analysisPresentation";

import LineBandChart from "./LineBandChart";
import NativeHistogramChart from "./NativeHistogramChart";
import RequestedBoxPlotChart from "./RequestedBoxPlotChart";
import RequestedHeatmapChart from "./RequestedHeatmapChart";
import RequestedLorenzChart from "./RequestedLorenzChart";
import ScatterPlot from "./ScatterPlot";
import SimpleLineChart from "./SimpleLineChart";

import {
  GapSummaryChart,
  GroupedSummaryChart,
  SimpleBarChart,
} from "./SummaryCharts";


export default function FindingChart({
  finding,
}: {
  finding:
    ReportFinding;
}) {
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

  const groupColumn =
    metricString(
      finding.metrics,
      "group_column"
    );

  const measureColumn =
    metricString(
      finding.metrics,
      "measure_column"
    ) ??
    metricString(
      finding.metrics,
      "value_column"
    );

  const valueColumn =
    metricString(
      finding.metrics,
      "value_column"
    ) ??
    measureColumn;


  switch (
    finding.chart_type
  ) {
    case "line":
      return (
        <SimpleLineChart
          data={
            finding.chart_data ??
            []
          }
          xLabel={
            friendlyVariableLabel(
              metricString(
                finding.metrics,
                "time_column"
              ) ??
              "Période"
            )
          }
          yLabel={
            friendlyVariableLabel(
              measureColumn ??
              "Valeur"
            )
          }
        />
      );


    case "bar":
      return (
        <SimpleBarChart
          finding={
            finding
          }
        />
      );


    case "line_band":
      return (
        <LineBandChart
          data={
            finding.chart_data ??
            []
          }
          yLabel={
            friendlyVariableLabel(
              measureColumn ??
              "Valeur"
            )
          }
        />
      );


    case "grouped_summary":
      return (
        <GroupedSummaryChart
          data={
            finding.chart_data ??
            []
          }
        />
      );


    case "scatter":
      return (
        <ScatterPlot
          data={
            finding.chart_data ??
            []
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
      );


    case "heatmap":
      return (
        <RequestedHeatmapChart
          data={
            finding.chart_data ??
            []
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
      );


    case "boxplot":
      return (
        <RequestedBoxPlotChart
          data={
            finding.chart_data ??
            []
          }
          groupLabel={
            friendlyVariableLabel(
              groupColumn ??
              "Groupe"
            )
          }
          valueLabel={
            friendlyVariableLabel(
              measureColumn ??
              "Valeur"
            )
          }
        />
      );


    case "histogram":
      return (
        <NativeHistogramChart
          data={
            finding.chart_data ??
            []
          }
          valueLabel={
            friendlyVariableLabel(
              valueColumn ??
              "Valeur"
            )
          }
        />
      );


    case "lorenz":
      return (
        <RequestedLorenzChart
          data={
            finding.chart_data ??
            []
          }
        />
      );


    case "distribution":
      return (
        <GapSummaryChart
          finding={
            finding
          }
        />
      );


    default:
      return (
        <div
          className={
            styles.chartEmpty
          }
        >
          Aucune visualisation
          détaillée disponible
          pour cette analyse.
        </div>
      );
  }
}

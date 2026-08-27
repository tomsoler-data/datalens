import type {
  DataQualityReportView,
  PreparationIssueSeverity,
} from "./preparationTypes";


export function preparationSeverityLabel(
  severity:
    PreparationIssueSeverity
): string {
  switch (
    severity
  ) {
    case "important":
      return "Important";

    case "moderate":
      return "Modéré";

    case "minor":
      return "Mineur";

    default:
      return "À examiner";
  }
}


export function preparationSeverityBorder(
  severity:
    PreparationIssueSeverity
): string {
  switch (
    severity
  ) {
    case "important":
      return "rgba(255, 142, 117, 0.24)";

    case "moderate":
      return "rgba(255, 187, 112, 0.22)";

    case "minor":
      return "rgba(126, 177, 255, 0.16)";

    default:
      return "rgba(255,255,255,0.08)";
  }
}


export function preparationQualityLabel(
  report:
    DataQualityReportView
): string {
  if (
    report.important_count >
    0
  ) {
    return "Attention requise";
  }


  if (
    report.moderate_count >
    0
  ) {
    return "À contrôler";
  }


  return "Satisfaisante";
}


export function formatBytes(
  bytes: number
): string {
  if (
    bytes <
    1024
  ) {
    return `${bytes} o`;
  }


  if (
    bytes <
    1024 * 1024
  ) {
    return `${(
      bytes /
      1024
    ).toFixed(
      1
    )} Ko`;
  }


  return `${(
    bytes /
    (
      1024 *
      1024
    )
  ).toFixed(
    1
  )} Mo`;
}

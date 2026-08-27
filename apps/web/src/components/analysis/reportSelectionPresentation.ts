import type {
  ReportSelectionItemView,
} from "./analysisTypes";

import type {
  RequestedAnalysisLifecycleView,
} from "./requestedAnalysisResolution";


export function requestedLifecycleStatusLabel(
  status:
    string |
    undefined
): string {
  const normalized =
    (
      status ??
      ""
    )
      .trim()
      .toLowerCase();


  if (
    normalized ===
    "ambiguous"
  ) {
    return "Ambigu\u00eb";
  }


  if (
    normalized ===
    "blocked"
  ) {
    return "Bloqu\u00e9e";
  }


  return "Non ex\u00e9cut\u00e9e";
}


export function requestedLifecycleReasons(
  lifecycle:
    RequestedAnalysisLifecycleView
): string[] {
  const warnings =
    Array.isArray(
      lifecycle.warnings
    )
      ? lifecycle.warnings.filter(
          (
            value
          ): value is string =>
            typeof value ===
              "string" &&
            value.trim().length >
              0
        )
      : [];


  if (
    warnings.length >
    0
  ) {
    return warnings.slice(
      0,
      2
    );
  }


  const limitations =
    Array.isArray(
      lifecycle.limitations
    )
      ? lifecycle.limitations.filter(
          (
            value
          ): value is string =>
            typeof value ===
              "string" &&
            value.trim().length >
              0
        )
      : [];


  if (
    limitations.length >
    0
  ) {
    return limitations.slice(
      0,
      2
    );
  }


  return [
    "La demande n'a pas pu produire de r\u00e9sultat analytique."
  ];
}


export function requestedLifecycleSource(
  lifecycle:
    RequestedAnalysisLifecycleView
): string {
  const filename =
    (
      lifecycle
        .source_filename ??
      ""
    ).trim();

  const locator =
    (
      lifecycle
        .source_locator ??
      ""
    ).trim();


  if (
    filename &&
    locator
  ) {
    return (
      `${filename} \u00b7 ${locator}`
    );
  }


  return (
    filename ||
    locator
  );
}


export function reportSourceLabel(
  sourceType:
    ReportSelectionItemView[
      "source_type"
    ]
): string {
  if (
    sourceType ===
    "initial_request"
  ) {
    return "Demande initiale";
  }


  if (
    sourceType ===
    "follow_up_prompt"
  ) {
    return "Question de suivi";
  }


  if (
    sourceType ===
    "document_request"
  ) {
    return "Demande du document";
  }


  return "Analyse automatique";
}

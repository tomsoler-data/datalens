import type {
  Dispatch,
  SetStateAction,
} from "react";

import type {
  ReportAvailableAnalysisDetailView,
  RequestedTimeGranularity,
} from "./analysisTypes";
import type { ReportRequestedFinding }
  from "../../app/types";


export type RequestedAnalysisLifecycleView = {
  request_id?:
    string;

  request_text?:
    string;

  request_order?:
    number;

  plan_status?:
    string;

  execution_status?:
    string;

  inferential_status?:
    string;

  warnings?:
    string[];

  limitations?:
    string[];

  source_filename?:
    string;

  source_locator?:
    string;

  source_evidence_quote?:
    string;
};


export function requestedLifecyclePayload(
  analysis:
    ReportAvailableAnalysisDetailView
): Record<
  string,
  unknown
> | null {
  const candidate = (
    analysis.pipeline_payload
  ) as unknown;


  if (
    candidate ===
      null ||
    typeof candidate !==
      "object" ||
    Array.isArray(
      candidate
    )
  ) {
    return null;
  }


  return candidate as Record<
    string,
    unknown
  >;
}

export function requestedFindingFromAvailableAnalysis(
  analysis:
    ReportAvailableAnalysisDetailView |
    null |
    undefined
): ReportRequestedFinding |
  null {
  if (
    !analysis
  ) {
    return null;
  }


  const payload =
    requestedLifecyclePayload(
      analysis
    );


  const candidate =
    payload?.[
      "requested_finding"
    ];


  if (
    candidate ===
      null ||
    candidate ===
      undefined ||
    typeof candidate !==
      "object" ||
    Array.isArray(
      candidate
    )
  ) {
    return null;
  }


  const record =
    candidate as Record<
      string,
      unknown
    >;


  if (
    typeof record[
      "analysis_id"
    ] !==
      "string"
  ) {
    return null;
  }


  return candidate as
    ReportRequestedFinding;
}


export function requestedLifecycleForAnalysis(
  analysis:
    ReportAvailableAnalysisDetailView
): RequestedAnalysisLifecycleView |
  null {
  const payload =
    requestedLifecyclePayload(
      analysis
    );


  if (
    payload ===
    null ||
    payload[
      "artifact_kind"
    ] !==
      "requested_analysis_lifecycle"
  ) {
    return null;
  }


  const lifecycle =
    payload[
      "request_lifecycle"
    ];


  if (
    lifecycle ===
      null ||
    typeof lifecycle !==
      "object" ||
    Array.isArray(
      lifecycle
    )
  ) {
    return null;
  }


  return (
    lifecycle
  ) as RequestedAnalysisLifecycleView;
}


export type RequestedRankingMetric =
  | "revenue"
  | "transaction_count";


type RequestedRankingResolutionAvailability = {
  requestId:
    string |
    null;

  isRankingRequest:
    boolean;

  revenue:
    boolean;

  transactionCount:
    boolean;
};


export function requestedRankingResolutionAvailability(
  analysis:
    ReportAvailableAnalysisDetailView
): RequestedRankingResolutionAvailability {
  const payload =
    requestedLifecyclePayload(
      analysis
    );

  const rawPlan =
    payload?.[
      "requested_plan"
    ];


  if (
    rawPlan ===
      null ||
    typeof rawPlan !==
      "object" ||
    Array.isArray(
      rawPlan
    )
  ) {
    return {
      requestId:
        null,

      isRankingRequest:
        false,

      revenue:
        false,

      transactionCount:
        false,
    };
  }


  const plan =
    rawPlan as Record<
      string,
      unknown
    >;


  const lifecycle =
    requestedLifecycleForAnalysis(
      analysis
    );


  const rawRequestId =
    typeof plan[
      "request_id"
    ] ===
      "string"
      ? plan[
          "request_id"
        ]
      : lifecycle
          ?.request_id;


  const requestId =
    typeof rawRequestId ===
      "string" &&
    rawRequestId.trim()
      ? rawRequestId.trim()
      : null;


  const kind =
    typeof plan[
      "kind"
    ] ===
      "string"
      ? plan[
          "kind"
        ]
          .trim()
          .toLowerCase()
      : "";


  const status =
    typeof plan[
      "status"
    ] ===
      "string"
      ? plan[
          "status"
        ]
          .trim()
          .toLowerCase()
      : "";


  const rawMatches =
    plan[
      "matched_columns"
    ];


  const concepts =
    new Set<
      string
    >();


  if (
    Array.isArray(
      rawMatches
    )
  ) {
    for (
      const rawMatch
      of rawMatches
    ) {
      if (
        rawMatch ===
          null ||
        typeof rawMatch !==
          "object" ||
        Array.isArray(
          rawMatch
        )
      ) {
        continue;
      }


      const candidate =
        rawMatch as Record<
          string,
          unknown
        >;

      const concept =
        candidate[
          "concept"
        ];


      if (
        typeof concept ===
          "string" &&
        concept.trim()
      ) {
        concepts.add(
          concept
            .trim()
            .toLowerCase()
        );
      }
    }
  }


  const hasProduct =
    concepts.has(
      "product_id"
    );


  return {
    requestId,

    isRankingRequest:
      status ===
        "ambiguous" &&
      (
        kind ===
          "top_products" ||
        kind ===
          "flop_products"
      ),

    revenue:
      hasProduct &&
      concepts.has(
        "amount"
      ),

    transactionCount:
      hasProduct &&
      (
        concepts.has(
          "transaction_id"
        ) ||
        concepts.has(
          "session_id"
        )
      ),
  };
}


type RequestedTimeSeriesResolutionAvailability = {
  requestId:
    string |
    null;

  isTimeSeriesRequest:
    boolean;

  executableInputs:
    boolean;
};


export function requestedTimeSeriesResolutionAvailability(
  analysis:
    ReportAvailableAnalysisDetailView
): RequestedTimeSeriesResolutionAvailability {
  const payload =
    requestedLifecyclePayload(
      analysis
    );


  const rawPlan =
    payload?.[
      "requested_plan"
    ];


  if (
    rawPlan ===
      null ||
    typeof rawPlan !==
      "object" ||
    Array.isArray(
      rawPlan
    )
  ) {
    return {
      requestId:
        null,

      isTimeSeriesRequest:
        false,

      executableInputs:
        false,
    };
  }


  const plan =
    rawPlan as Record<
      string,
      unknown
    >;


  const rawRequestId =
    plan[
      "request_id"
    ];


  const requestId =
    typeof rawRequestId ===
      "string" &&
    rawRequestId.trim()
      ? rawRequestId.trim()
      : null;


  const status =
    typeof plan[
      "status"
    ] ===
      "string"
      ? (
          plan[
            "status"
          ] as string
        )
          .trim()
          .toLowerCase()
      : "";


  const kind =
    typeof plan[
      "kind"
    ] ===
      "string"
      ? (
          plan[
            "kind"
          ] as string
        )
          .trim()
          .toLowerCase()
      : "";


  const rawMatches =
    plan[
      "matched_columns"
    ];


  const concepts =
    new Set<
      string
    >();


  if (
    Array.isArray(
      rawMatches
    )
  ) {
    for (
      const rawMatch
      of rawMatches
    ) {
      if (
        rawMatch ===
          null ||
        typeof rawMatch !==
          "object" ||
        Array.isArray(
          rawMatch
        )
      ) {
        continue;
      }


      const candidate =
        rawMatch as Record<
          string,
          unknown
        >;


      const concept =
        candidate[
          "concept"
        ];


      if (
        typeof concept ===
          "string" &&
        concept.trim()
      ) {
        concepts.add(
          concept
            .trim()
            .toLowerCase()
        );
      }
    }
  }


  return {
    requestId,

    isTimeSeriesRequest:
      status ===
        "ambiguous" &&
      kind ===
        "revenue_moving_average",

    executableInputs:
      concepts.has(
        "amount"
      ) &&
      concepts.has(
        "time"
      ),
  };
}



export type FollowUpRequestedAnalysisRouteView = {
  workflow_id:
    string;

  objective:
    string;

  route_kind:
    "requested_analysis" |
    "ai_native";

  analysis_id:
    string |
    null;

  request_id:
    string |
    null;

  kind:
    string |
    null;

  plan_status:
    string |
    null;

  source_type:
    string |
    null;

  api_version:
    string;
};


export async function routeFollowUpRequestedAnalysis({
  apiUrl,
  workflowId,
  objective,
}: {
  apiUrl:
    string;

  workflowId:
    string;

  objective:
    string;
}): Promise<
  FollowUpRequestedAnalysisRouteView
> {
  const response =
    await fetch(
      `${apiUrl}/analysis/requested/route-follow-up`,
      {
        method:
          "POST",

        headers: {
          "Content-Type":
            "application/json",
        },

        body:
          JSON.stringify(
            {
              workflow_id:
                workflowId,

              objective,
            }
          ),
      }
    );


  let payload:
    unknown =
      null;


  try {
    payload =
      await response.json();
  } catch {
    payload =
      null;
  }


  if (
    !response.ok
  ) {
    let message =
      "Le routage de la nouvelle demande a ?chou?.";


    if (
      payload !==
        null &&
      typeof payload ===
        "object" &&
      !Array.isArray(
        payload
      )
    ) {
      const payloadRecord =
        payload as Record<
          string,
          unknown
        >;

      const detail =
        payloadRecord[
          "detail"
        ];


      if (
        typeof detail ===
          "string" &&
        detail.trim()
      ) {
        message =
          detail.trim();
      } else if (
        detail !==
          null &&
        typeof detail ===
          "object" &&
        !Array.isArray(
          detail
        )
      ) {
        const detailRecord =
          detail as Record<
            string,
            unknown
          >;

        const detailMessage =
          detailRecord[
            "message"
          ];


        if (
          typeof detailMessage ===
            "string" &&
          detailMessage.trim()
        ) {
          message =
            detailMessage.trim();
        }
      }
    }


    throw new Error(
      message
    );
  }


  if (
    payload ===
      null ||
    typeof payload !==
      "object" ||
    Array.isArray(
      payload
    )
  ) {
    throw new Error(
      "Le routeur de suivi a renvoy? une r?ponse invalide."
    );
  }


  const route =
    payload as Record<
      string,
      unknown
    >;


  const routeKind =
    route[
      "route_kind"
    ];


  if (
    routeKind !==
      "requested_analysis" &&
    routeKind !==
      "ai_native"
  ) {
    throw new Error(
      "Le routeur de suivi a renvoy? un type de route inconnu."
    );
  }


  return (
    payload
  ) as FollowUpRequestedAnalysisRouteView;
}


type RequestedAnalysisResolutionSession = {
  readonly workflow_id?:
    string |
    null;
};


type RequestedAnalysisResolutionDependencies = {
  apiUrl:
    string;

  preparationSession:
    RequestedAnalysisResolutionSession |
    null;

  setRequestedResolutionLoadingId:
    Dispatch<
      SetStateAction<
        string |
        null
      >
    >;

  setRequestedResolutionErrors:
    Dispatch<
      SetStateAction<
        Record<
          string,
          string
        >
      >
    >;

  refreshReportSelection:
    (
      workflowId:
        string
    ) => Promise<
      unknown
    >;
};


export function createRequestedAnalysisResolutionHandlers({
  apiUrl,
  preparationSession,
  setRequestedResolutionLoadingId,
  setRequestedResolutionErrors,
  refreshReportSelection,
}: RequestedAnalysisResolutionDependencies) {
  async function handleResolveRequestedRanking(
    analysis:
      ReportAvailableAnalysisDetailView,

    rankingMetric:
      RequestedRankingMetric
  ): Promise<
    void
  > {
    const workflowId =
      preparationSession
        ?.workflow_id;


    const availability =
      requestedRankingResolutionAvailability(
        analysis
      );


    const requestId =
      availability
        .requestId;


    if (
      !workflowId
    ) {
      setRequestedResolutionErrors(
        (
          current
        ) => ({
          ...current,

          [
            analysis.analysis_id
          ]:
            "Aucun workflow Preparation actif n'est disponible.",
        })
      );

      return;
    }


    if (
      !requestId
    ) {
      setRequestedResolutionErrors(
        (
          current
        ) => ({
          ...current,

          [
            analysis.analysis_id
          ]:
            "L'identifiant server-owned de la demande est indisponible.",
        })
      );

      return;
    }


    const metricAvailable =
      rankingMetric ===
        "revenue"
        ? availability
            .revenue
        : availability
            .transactionCount;


    if (
      !availability
        .isRankingRequest ||
      !metricAvailable
    ) {
      setRequestedResolutionErrors(
        (
          current
        ) => ({
          ...current,

          [
            analysis.analysis_id
          ]:
            "Cette métrique n'est pas disponible dans le plan analytique résolu.",
        })
      );

      return;
    }


    setRequestedResolutionLoadingId(
      analysis.analysis_id
    );


    setRequestedResolutionErrors(
      (
        current
      ) => {
        const next = {
          ...current,
        };

        delete next[
          analysis.analysis_id
        ];

        return next;
      }
    );


    try {
      const response =
        await fetch(
          `${apiUrl}/analysis/requested/resolve`,
          {
            method:
              "POST",

            headers: {
              "Content-Type":
                "application/json",
            },

            body:
              JSON.stringify(
                {
                  workflow_id:
                    workflowId,

                  request_id:
                    requestId,

                  resolution: {
                    resolution_type:
                      "ranking_metric",

                    ranking_metric:
                      rankingMetric,
                  },
                }
              ),
          }
        );


      let payload:
        unknown =
          null;


      try {
        payload =
          await response.json();
      } catch {
        payload =
          null;
      }


      if (
        !response.ok
      ) {
        let message =
          "La clarification de la demande a été refusée.";


        if (
          payload !==
            null &&
          typeof payload ===
            "object" &&
          !Array.isArray(
            payload
          )
        ) {
          const payloadRecord =
            payload as Record<
              string,
              unknown
            >;

          const detail =
            payloadRecord[
              "detail"
            ];


          if (
            typeof detail ===
              "string" &&
            detail.trim()
          ) {
            message =
              detail.trim();
          } else if (
            detail !==
              null &&
            typeof detail ===
              "object" &&
            !Array.isArray(
              detail
            )
          ) {
            const detailRecord =
              detail as Record<
                string,
                unknown
              >;

            const detailMessage =
              detailRecord[
                "message"
              ];


            if (
              typeof detailMessage ===
                "string" &&
              detailMessage.trim()
            ) {
              message =
                detailMessage.trim();
            }
          }
        }


        throw new Error(
          message
        );
      }


      const refreshed =
        await refreshReportSelection(
          workflowId
        );


      if (
        refreshed ===
          null
      ) {
        throw new Error(
          (
            "La demande a été résolue côté serveur, "
            +
            "mais l'état du rapport n'a pas pu être actualisé."
          )
        );
      }
    } catch (
      caughtError
    ) {
      setRequestedResolutionErrors(
        (
          current
        ) => ({
          ...current,

          [
            analysis.analysis_id
          ]:
            caughtError
              instanceof Error
              ? caughtError.message
              : "La clarification de la demande a échoué.",
        })
      );
    } finally {
      setRequestedResolutionLoadingId(
        (
          current
        ) =>
          current ===
            analysis.analysis_id
            ? null
            : current
      );
    }
  }


  async function handleReconfigureRequestedTimeSeries(
    analysis:
      ReportAvailableAnalysisDetailView,

    timeGranularity:
      RequestedTimeGranularity,

    movingAverageWindow:
      number
  ): Promise<
    void
  > {
    const workflowId =
      preparationSession
        ?.workflow_id;


    const lifecycle =
      requestedLifecycleForAnalysis(
        analysis
      );


    const lifecycleRequestId =
      lifecycle
        ?.request_id;


    const fallbackAvailability =
      requestedTimeSeriesResolutionAvailability(
        analysis
      );


    const requestId =
      typeof lifecycleRequestId ===
        "string" &&
      lifecycleRequestId.trim()
        ? lifecycleRequestId.trim()
        : fallbackAvailability
            .requestId;


    if (
      !workflowId
    ) {
      setRequestedResolutionErrors(
        (
          current
        ) => ({
          ...current,

          [
            analysis.analysis_id
          ]:
            "Aucun workflow Preparation actif n'est disponible.",
        })
      );

      return;
    }


    if (
      !requestId
    ) {
      setRequestedResolutionErrors(
        (
          current
        ) => ({
          ...current,

          [
            analysis.analysis_id
          ]:
            "L'identifiant server-owned de la demande est indisponible.",
        })
      );

      return;
    }


    const allowedGranularities:
      RequestedTimeGranularity[] =
        [
          "day",
          "week",
          "month",
          "quarter",
          "year",
        ];


    if (
      !allowedGranularities.includes(
        timeGranularity
      )
    ) {
      setRequestedResolutionErrors(
        (
          current
        ) => ({
          ...current,

          [
            analysis.analysis_id
          ]:
            "La granularite temporelle selectionnee n'est pas prise en charge.",
        })
      );

      return;
    }


    if (
      !Number.isInteger(
        movingAverageWindow
      ) ||
      movingAverageWindow <
        1
    ) {
      setRequestedResolutionErrors(
        (
          current
        ) => ({
          ...current,

          [
            analysis.analysis_id
          ]:
            "La fenetre de moyenne mobile doit etre un entier superieur ou egal a 1.",
        })
      );

      return;
    }


    setRequestedResolutionLoadingId(
      analysis.analysis_id
    );


    setRequestedResolutionErrors(
      (
        current
      ) => {
        const next = {
          ...current,
        };


        delete next[
          analysis.analysis_id
        ];


        return next;
      }
    );


    try {
      const response =
        await fetch(
          `${apiUrl}/analysis/requested/reconfigure`,
          {
            method:
              "POST",

            headers: {
              "Content-Type":
                "application/json",
            },

            body:
              JSON.stringify(
                {
                  workflow_id:
                    workflowId,

                  request_id:
                    requestId,

                  resolution: {
                    resolution_type:
                      "time_series_parameters",

                    time_granularity:
                      timeGranularity,

                    moving_average_window:
                      movingAverageWindow,
                  },
                }
              ),
          }
        );


      let payload:
        unknown =
          null;


      try {
        payload =
          await response.json();
      } catch {
        payload =
          null;
      }


      if (
        !response.ok
      ) {
        let message =
          "La reconfiguration temporelle a ete refusee.";


        if (
          payload !==
            null &&
          typeof payload ===
            "object" &&
          !Array.isArray(
            payload
          )
        ) {
          const payloadRecord =
            payload as Record<
              string,
              unknown
            >;


          const detail =
            payloadRecord[
              "detail"
            ];


          if (
            typeof detail ===
              "string" &&
            detail.trim()
          ) {
            message =
              detail.trim();
          } else if (
            detail !==
              null &&
            typeof detail ===
              "object" &&
            !Array.isArray(
              detail
            )
          ) {
            const detailRecord =
              detail as Record<
                string,
                unknown
              >;


            const detailMessage =
              detailRecord[
                "message"
              ];


            if (
              typeof detailMessage ===
                "string" &&
              detailMessage.trim()
            ) {
              message =
                detailMessage.trim();
            }
          }
        }


        throw new Error(
          message
        );
      }


      const refreshed =
        await refreshReportSelection(
          workflowId
        );


      if (
        refreshed ===
          null
      ) {
        throw new Error(
          (
            "La demande a ete reconfiguree cote serveur, "
            +
            "mais l'etat du rapport n'a pas pu etre actualise."
          )
        );
      }
    } catch (
      caughtError
    ) {
      setRequestedResolutionErrors(
        (
          current
        ) => ({
          ...current,

          [
            analysis.analysis_id
          ]:
            caughtError
              instanceof Error
              ? caughtError.message
              : "La reconfiguration temporelle a echoue.",
        })
      );
    } finally {
      setRequestedResolutionLoadingId(
        (
          current
        ) =>
          current ===
            analysis.analysis_id
            ? null
            : current
      );
    }
  }


  async function handleResolveRequestedTimeSeries(
    analysis:
      ReportAvailableAnalysisDetailView,

    timeGranularity:
      RequestedTimeGranularity,

    movingAverageWindow:
      number
  ): Promise<
    void
  > {
    const workflowId =
      preparationSession
        ?.workflow_id;


    const availability =
      requestedTimeSeriesResolutionAvailability(
        analysis
      );


    const requestId =
      availability
        .requestId;


    if (
      !workflowId
    ) {
      setRequestedResolutionErrors(
        (
          current
        ) => ({
          ...current,

          [
            analysis.analysis_id
          ]:
            "Aucun workflow Preparation actif n'est disponible.",
        })
      );

      return;
    }


    if (
      !requestId
    ) {
      setRequestedResolutionErrors(
        (
          current
        ) => ({
          ...current,

          [
            analysis.analysis_id
          ]:
            "L'identifiant server-owned de la demande est indisponible.",
        })
      );

      return;
    }


    if (
      !availability
        .isTimeSeriesRequest ||
      !availability
        .executableInputs
    ) {
      setRequestedResolutionErrors(
        (
          current
        ) => ({
          ...current,

          [
            analysis.analysis_id
          ]:
            "Les variables temporelle et monetaire requises ne sont pas disponibles dans le plan server-owned.",
        })
      );

      return;
    }


    const allowedGranularities:
      RequestedTimeGranularity[] =
        [
          "day",
          "week",
          "month",
          "quarter",
          "year",
        ];


    if (
      !allowedGranularities.includes(
        timeGranularity
      )
    ) {
      setRequestedResolutionErrors(
        (
          current
        ) => ({
          ...current,

          [
            analysis.analysis_id
          ]:
            "La granularite temporelle selectionnee n'est pas prise en charge.",
        })
      );

      return;
    }


    if (
      !Number.isInteger(
        movingAverageWindow
      ) ||
      movingAverageWindow <
        1
    ) {
      setRequestedResolutionErrors(
        (
          current
        ) => ({
          ...current,

          [
            analysis.analysis_id
          ]:
            "La fenetre de moyenne mobile doit etre un entier superieur ou egal a 1.",
        })
      );

      return;
    }


    setRequestedResolutionLoadingId(
      analysis.analysis_id
    );


    setRequestedResolutionErrors(
      (
        current
      ) => {
        const next = {
          ...current,
        };


        delete next[
          analysis.analysis_id
        ];


        return next;
      }
    );


    try {
      const response =
        await fetch(
          `${apiUrl}/analysis/requested/resolve`,
          {
            method:
              "POST",

            headers: {
              "Content-Type":
                "application/json",
            },

            body:
              JSON.stringify(
                {
                  workflow_id:
                    workflowId,

                  request_id:
                    requestId,

                  resolution: {
                    resolution_type:
                      "time_series_parameters",

                    time_granularity:
                      timeGranularity,

                    moving_average_window:
                      movingAverageWindow,
                  },
                }
              ),
          }
        );


      let payload:
        unknown =
          null;


      try {
        payload =
          await response.json();
      } catch {
        payload =
          null;
      }


      if (
        !response.ok
      ) {
        let message =
          "La clarification temporelle a ete refusee.";


        if (
          payload !==
            null &&
          typeof payload ===
            "object" &&
          !Array.isArray(
            payload
          )
        ) {
          const payloadRecord =
            payload as Record<
              string,
              unknown
            >;


          const detail =
            payloadRecord[
              "detail"
            ];


          if (
            typeof detail ===
              "string" &&
            detail.trim()
          ) {
            message =
              detail.trim();
          } else if (
            detail !==
              null &&
            typeof detail ===
              "object" &&
            !Array.isArray(
              detail
            )
          ) {
            const detailRecord =
              detail as Record<
                string,
                unknown
              >;


            const detailMessage =
              detailRecord[
                "message"
              ];


            if (
              typeof detailMessage ===
                "string" &&
              detailMessage.trim()
            ) {
              message =
                detailMessage.trim();
            }
          }
        }


        throw new Error(
          message
        );
      }


      const refreshed =
        await refreshReportSelection(
          workflowId
        );


      if (
        refreshed ===
          null
      ) {
        throw new Error(
          (
            "La demande a ete resolue cote serveur, "
            +
            "mais l'etat du rapport n'a pas pu etre actualise."
          )
        );
      }
    } catch (
      caughtError
    ) {
      setRequestedResolutionErrors(
        (
          current
        ) => ({
          ...current,

          [
            analysis.analysis_id
          ]:
            caughtError
              instanceof Error
              ? caughtError.message
              : "La clarification temporelle a echoue.",
        })
      );
    } finally {
      setRequestedResolutionLoadingId(
        (
          current
        ) =>
          current ===
            analysis.analysis_id
            ? null
            : current
      );
    }
  }


  return {
    handleResolveRequestedRanking,
    handleReconfigureRequestedTimeSeries,
    handleResolveRequestedTimeSeries,
  };
}


export function requestedLifecycleOrder(
  analysis:
    ReportAvailableAnalysisDetailView
): number {
  const lifecycle =
    requestedLifecycleForAnalysis(
      analysis
    );

  const lifecycleOrder =
    lifecycle
      ?.request_order;


  if (
    typeof lifecycleOrder ===
      "number" &&
    Number.isFinite(
      lifecycleOrder
    )
  ) {
    return lifecycleOrder;
  }


  const payload =
    requestedLifecyclePayload(
      analysis
    );

  const payloadOrder =
    payload?.[
      "request_order"
    ];


  if (
    typeof payloadOrder ===
      "number" &&
    Number.isFinite(
      payloadOrder
    )
  ) {
    return payloadOrder;
  }


  return Number.MAX_SAFE_INTEGER;
}

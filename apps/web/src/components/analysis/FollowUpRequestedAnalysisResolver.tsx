"use client";


import type {
  ReportAvailableAnalysisDetailView,
  RequestedTimeGranularity,
} from "./analysisTypes";

import {
  requestedLifecycleForAnalysis,
  requestedTimeSeriesResolutionAvailability,
} from "./requestedAnalysisResolution";


type FollowUpRequestedAnalysisResolverProps = {
  analysis:
    ReportAvailableAnalysisDetailView;

  loading:
    boolean;

  error:
    string |
    null;

  onResolveTimeSeries: (
    analysis:
      ReportAvailableAnalysisDetailView,

    timeGranularity:
      RequestedTimeGranularity,

    movingAverageWindow:
      number
  ) => Promise<void>;
};


export default function FollowUpRequestedAnalysisResolver({
  analysis,
  loading,
  error,
  onResolveTimeSeries,
}: FollowUpRequestedAnalysisResolverProps) {
  const lifecycle =
    requestedLifecycleForAnalysis(
      analysis
    );

  const availability =
    requestedTimeSeriesResolutionAvailability(
      analysis
    );


  if (
    lifecycle ===
      null ||
    !availability
      .isTimeSeriesRequest ||
    !availability
      .executableInputs ||
    !availability
      .requestId
  ) {
    return null;
  }


  const requestText =
    (
      lifecycle
        .request_text ??
      analysis.objective
    )
      .trim();


  return (
    <article
      style={{
        marginTop:
          "16px",

        padding:
          "16px",

        border:
          "1px solid rgba(232, 184, 97, 0.28)",

        borderRadius:
          "12px",

        background:
          "linear-gradient(135deg, rgba(232, 184, 97, 0.065), rgba(8, 18, 29, 0.52))",
      }}
      aria-live="polite"
    >
      <span
        style={{
          display:
            "inline-flex",

          alignItems:
            "center",

          minHeight:
            "24px",

          padding:
            "0 8px",

          border:
            "1px solid rgba(232, 184, 97, 0.28)",

          borderRadius:
            "999px",

          color:
            "#e8c77c",

          background:
            "rgba(232, 184, 97, 0.06)",

          fontSize:
            "0.61rem",

          fontWeight:
            760,

          letterSpacing:
            "0.055em",

          textTransform:
            "uppercase",
        }}
      >
        Param?tres requis
      </span>


      <strong
        style={{
          display:
            "block",

          marginTop:
            "10px",

          color:
            "#edf3fb",

          fontSize:
            "0.84rem",

          lineHeight:
            1.5,
        }}
      >
        {
          requestText
        }
      </strong>


      <p
        style={{
          margin:
            "6px 0 0",

          color:
            "#9eafc3",

          fontSize:
            "0.72rem",

          lineHeight:
            1.6,
        }}
      >
        DataLens a reconnu une analyse temporelle d?terministe.
        Choisissez la p?riode et la fen?tre de moyenne mobile
        avant le calcul. Aucun dataset ni aucune colonne ne sont
        s?lectionn?s par le navigateur.
      </p>


      <form
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
                  "time_granularity"
                ) ??
                ""
              );


            const allowed:
              RequestedTimeGranularity[] =
                [
                  "day",
                  "week",
                  "month",
                  "quarter",
                  "year",
                ];


            if (
              !allowed.includes(
                rawGranularity as
                  RequestedTimeGranularity
              )
            ) {
              return;
            }


            const windowValue =
              Number(
                formData.get(
                  "moving_average_window"
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


            void onResolveTimeSeries(
              analysis,
              rawGranularity as
                RequestedTimeGranularity,
              windowValue
            );
          }
        }
      >
        <div
          style={{
            display:
              "grid",

            gridTemplateColumns:
              "minmax(160px, 220px) minmax(130px, 180px)",

            gap:
              "10px",

            marginTop:
              "14px",

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
                "#b5c2d3",

              fontSize:
                "0.67rem",
            }}
          >
            P?riode

            <select
              name="time_granularity"
              defaultValue="month"
              disabled={
                loading
              }
              style={{
                minHeight:
                  "38px",

                padding:
                  "0 9px",

                border:
                  "1px solid rgba(116, 177, 255, 0.24)",

                borderRadius:
                  "8px",

                background:
                  "rgba(8, 23, 34, 0.90)",

                color:
                  "#e3ecf7",
              }}
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
                Ann?e
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
                "#b5c2d3",

              fontSize:
                "0.67rem",
            }}
          >
            Moyenne mobile

            <input
              name="moving_average_window"
              type="number"
              min={
                1
              }
              step={
                1
              }
              defaultValue={
                3
              }
              required
              disabled={
                loading
              }
              style={{
                minHeight:
                  "38px",

                padding:
                  "0 9px",

                border:
                  "1px solid rgba(232, 184, 97, 0.32)",

                borderRadius:
                  "8px",

                background:
                  "rgba(20, 20, 17, 0.88)",

                color:
                  "#f0c979",
              }}
            />
          </label>
        </div>


        <button
          type="submit"
          disabled={
            loading
          }
          style={{
            minHeight:
              "38px",

            marginTop:
              "11px",

            padding:
              "0 13px",

            border:
              "1px solid rgba(232, 184, 97, 0.32)",

            borderRadius:
              "8px",

            background:
              "rgba(232, 184, 97, 0.08)",

            color:
              "#f1d18c",

            cursor:
              loading
                ? "wait"
                : "pointer",

            opacity:
              loading
                ? 0.65
                : 1,

            fontSize:
              "0.7rem",

            fontWeight:
              700,
          }}
        >
          {
            loading
              ? "Calcul en cours?"
              : "Calculer l?analyse"
          }
        </button>
      </form>


      {
        error
          ? (
              <div
                role="alert"
                style={{
                  marginTop:
                    "9px",

                  padding:
                    "9px 10px",

                  border:
                    "1px solid rgba(226, 112, 112, 0.20)",

                  borderRadius:
                    "8px",

                  background:
                    "rgba(137, 48, 48, 0.06)",

                  color:
                    "#efaaaa",

                  fontSize:
                    "0.68rem",
                }}
              >
                {
                  error
                }
              </div>
            )
          : null
      }
    </article>
  );
}

import type {
  ReportChartDatum,
} from "../../../app/types";

import styles from "../../../app/page.module.css";

import {
  datumLabel,
  datumNumber,
  formatChartNumber,
  formatDecimal,
} from "../analysisPresentation";


export default function RequestedBarChart({
  data,
  categoryLabel,
  valueLabel,
}: {
  data:
    ReportChartDatum[];

  categoryLabel:
    string;

  valueLabel:
    string;
}) {
  const points =
    data
      .map(
        (
          datum,
          index
        ) => {
          const label =
            datumLabel(
              datum,
              "category"
            ) ??
            datumLabel(
              datum,
              "group"
            ) ??
            datumLabel(
              datum,
              "label"
            ) ??
            `Catégorie ${index + 1}`;


          const value =
            datumNumber(
              datum,
              "value"
            );


          if (
            value ===
            null
          ) {
            return null;
          }


          return {
            label,
            value,
          };
        }
      )
      .filter(
        (
          point
        ): point is {
          label:
            string;

          value:
            number;
        } =>
          point !==
          null
      );


  if (
    points.length ===
    0
  ) {
    return (
      <div
        className={
          styles.chartEmpty
        }
      >
        Aucune catégorie exploitable.
      </div>
    );
  }


  const maxValue =
    Math.max(
      ...points.map(
        (
          point
        ) =>
          Math.abs(
            point.value
          )
      ),
      1
    );


  return (
    <div
      className={
        styles.technicalReasons
      }
    >
      {
        points.map(
          (
            point,
            index
          ) => (
            <div
              key={
                `${point.label}-${index}`
              }
              className={
                styles.summaryItem
              }
              title={
                `${categoryLabel}: ${point.label} · ${valueLabel}: ${formatDecimal(
                  point.value
                )}`
              }
            >
              <span>
                {
                  `${categoryLabel} · ${point.label}`
                }
              </span>

              <strong>
                {
                  formatChartNumber(
                    point.value
                  )
                }
              </strong>

              <div
                style={{
                  width:
                    "100%",

                  height:
                    "9px",

                  borderRadius:
                    "999px",

                  overflow:
                    "hidden",

                  background:
                    "rgba(255,255,255,0.08)",
                }}
                aria-hidden="true"
              >
                <div
                  style={{
                    width:
                      `${Math.max(
                        2,
                        (
                          Math.abs(
                            point.value
                          ) /
                          maxValue
                        ) *
                          100
                      )}%`,

                    height:
                      "100%",

                    borderRadius:
                      "999px",

                    background:
                      "currentColor",

                    opacity:
                      0.72,
                  }}
                />
              </div>
            </div>
          )
        )
      }
    </div>
  );
}

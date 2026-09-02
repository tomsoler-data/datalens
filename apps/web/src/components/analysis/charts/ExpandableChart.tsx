"use client";

import {
  useEffect,
  useState,
} from "react";

import type {
  ReactNode,
} from "react";


export default function ExpandableChart({
  title,
  children,
}: {
  title:
    string;

  children:
    ReactNode;
}) {
  const [
    expanded,
    setExpanded,
  ] = useState(
    false
  );


  useEffect(
    () => {
      if (
        !expanded
      ) {
        return;
      }


      const previousOverflow =
        document.body.style.overflow;


      document.body.style.overflow =
        "hidden";


      function handleKeyDown(
        event:
          KeyboardEvent
      ) {
        if (
          event.key ===
          "Escape"
        ) {
          setExpanded(
            false
          );
        }
      }


      window.addEventListener(
        "keydown",
        handleKeyDown
      );


      return () => {
        document.body.style.overflow =
          previousOverflow;

        window.removeEventListener(
          "keydown",
          handleKeyDown
        );
      };
    },
    [
      expanded,
    ]
  );


  return (
    <>
      <div
        style={{
          position:
            "relative",

          width:
            "100%",

          maxWidth:
            "1000px",

          margin:
            "0 auto",
        }}
      >
        <div
          style={{
            display:
              "flex",

            justifyContent:
              "flex-end",

            marginBottom:
              "8px",
          }}
        >
          <button
            type="button"
            aria-expanded={
              expanded
            }
            onClick={
              () =>
                setExpanded(
                  true
                )
            }
            style={{
              display:
                "inline-flex",

              alignItems:
                "center",

              justifyContent:
                "center",

              minHeight:
                "32px",

              padding:
                "0 10px",

              border:
                "1px solid rgba(126, 177, 255, 0.18)",

              borderRadius:
                "9px",

              color:
                "inherit",

              background:
                "rgba(126, 177, 255, 0.045)",

              font:
                "inherit",

              fontSize:
                "0.7rem",

              fontWeight:
                700,

              cursor:
                "pointer",
            }}
          >
            Agrandir ↗
          </button>
        </div>


        {
          children
        }
      </div>


      {
        expanded
          ? (
              <div
                role="dialog"
                aria-modal="true"
                aria-label={
                  `Graphique agrandi : ${title}`
                }
                onMouseDown={
                  (
                    event
                  ) => {
                    if (
                      event.target ===
                      event.currentTarget
                    ) {
                      setExpanded(
                        false
                      );
                    }
                  }
                }
                style={{
                  position:
                    "fixed",

                  inset:
                    0,

                  zIndex:
                    1000,

                  display:
                    "grid",

                  placeItems:
                    "center",

                  padding:
                    "3vh 3vw",

                  background:
                    "rgba(2, 8, 18, 0.86)",

                  backdropFilter:
                    "blur(14px)",
                }}
              >
                <section
                  style={{
                    width:
                      "min(1500px, 94vw)",

                    maxHeight:
                      "92vh",

                    display:
                      "grid",

                    gridTemplateRows:
                      "auto minmax(0, 1fr)",

                    overflow:
                      "hidden",

                    border:
                      "1px solid rgba(126, 177, 255, 0.22)",

                    borderRadius:
                      "18px",

                    background:
                      "linear-gradient(180deg, rgba(12, 25, 44, 0.99), rgba(7, 15, 28, 0.99))",

                    boxShadow:
                      "0 28px 90px rgba(0, 0, 0, 0.48)",
                  }}
                >
                  <header
                    style={{
                      display:
                        "flex",

                      alignItems:
                        "center",

                      justifyContent:
                        "space-between",

                      gap:
                        "18px",

                      padding:
                        "14px 16px",

                      borderBottom:
                        "1px solid rgba(255,255,255,0.07)",
                    }}
                  >
                    <div
                      style={{
                        minWidth:
                          0,
                      }}
                    >
                      <span
                        style={{
                          display:
                            "block",

                          marginBottom:
                            "4px",

                          fontSize:
                            "0.62rem",

                          fontWeight:
                            800,

                          letterSpacing:
                            "0.08em",

                          textTransform:
                            "uppercase",

                          opacity:
                            0.48,
                        }}
                      >
                        Visualisation agrandie
                      </span>

                      <strong
                        style={{
                          display:
                            "block",

                          overflow:
                            "hidden",

                          textOverflow:
                            "ellipsis",

                          whiteSpace:
                            "nowrap",

                          fontSize:
                            "0.94rem",
                        }}
                      >
                        {
                          title
                        }
                      </strong>
                    </div>


                    <button
                      type="button"
                      autoFocus
                      onClick={
                        () =>
                          setExpanded(
                            false
                          )
                      }
                      aria-label="Fermer le graphique agrandi"
                      style={{
                        flex:
                          "0 0 auto",

                        minHeight:
                          "34px",

                        padding:
                          "0 11px",

                        border:
                          "1px solid rgba(255,255,255,0.10)",

                        borderRadius:
                          "9px",

                        color:
                          "inherit",

                        background:
                          "rgba(255,255,255,0.035)",

                        font:
                          "inherit",

                        fontSize:
                          "0.7rem",

                        fontWeight:
                          700,

                        cursor:
                          "pointer",
                      }}
                    >
                      Fermer ×
                    </button>
                  </header>


                  <div
                    style={{
                      minHeight:
                        0,

                      overflow:
                        "auto",

                      padding:
                        "18px",

                      display:
                        "grid",

                      alignItems:
                        "center",
                    }}
                  >
                    <div
                      style={{
                        width:
                          "100%",

                        minWidth:
                          "760px",
                      }}
                    >
                      {
                        children
                      }
                    </div>
                  </div>
                </section>
              </div>
            )
          : null
      }
    </>
  );
}

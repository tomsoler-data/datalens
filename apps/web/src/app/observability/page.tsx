import {
  Suspense,
} from "react";

import ObservabilityClient from "./ObservabilityClient";


function ObservabilityLoading() {
  return (
    <main
      style={{
        minHeight:
          "100vh",

        display:
          "grid",

        placeItems:
          "center",

        padding:
          "24px",

        color:
          "#e9f0fb",

        background:
          "#07101d",
      }}
    >
      <div
        style={{
          padding:
            "18px 20px",

          border:
            "1px solid rgba(255, 255, 255, 0.075)",

          borderRadius:
            "14px",

          background:
            "rgba(255, 255, 255, 0.022)",

          fontSize:
            "0.76rem",

          opacity:
            0.7,
        }}
      >
        Chargement de l’observabilité…
      </div>
    </main>
  );
}


export default function ObservabilityPage() {
  return (
    <Suspense
      fallback={
        <ObservabilityLoading />
      }
    >
      <ObservabilityClient />
    </Suspense>
  );
}
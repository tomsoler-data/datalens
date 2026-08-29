import {
  Suspense,
} from "react";

import ModelLabClient
  from "./ModelLabClient";

import styles
  from "./modelLab.module.css";


function ModelLabLoading() {
  return (
    <main
      className={
        styles.loadingPage
      }
    >
      <div
        className={
          styles.loadingCard
        }
      >
        <span
          className={
            styles.loadingDot
          }
          aria-hidden="true"
        />

        <span>
          Chargement du Model Lab…
        </span>
      </div>
    </main>
  );
}


export default function ModelLabPage() {
  return (
    <Suspense
      fallback={
        <ModelLabLoading />
      }
    >
      <ModelLabClient />
    </Suspense>
  );
}

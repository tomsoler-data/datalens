import styles from "../../app/page.module.css";


export default function PreparationResolvedStagePanel({
  eyebrow,
  title,
  description,
  skipped,
}: {
  eyebrow:
    string;

  title:
    string;

  description:
    string;

  skipped:
    boolean;
}) {
  return (
    <section
      style={{
        marginTop:
          "18px",

        padding:
          "18px",

        border:
          skipped
            ? "1px solid rgba(255,255,255,0.07)"
            : "1px solid rgba(122,203,160,0.16)",

        borderRadius:
          "14px",

        background:
          skipped
            ? "rgba(255,255,255,0.008)"
            : "linear-gradient(180deg, rgba(122,203,160,0.045), rgba(255,255,255,0.008))",
      }}
    >
      <div
        style={{
          display:
            "flex",

          alignItems:
            "flex-start",

          justifyContent:
            "space-between",

          gap:
            "16px",
        }}
      >
        <div>
          <span
            className={
              styles.eyebrow
            }
          >
            {
              eyebrow
            }
          </span>

          <h3
            style={{
              margin:
                "7px 0 0",

              fontSize:
                "1rem",
            }}
          >
            {
              title
            }
          </h3>

          <p
            style={{
              margin:
                "7px 0 0",

              maxWidth:
                "780px",

              fontSize:
                "0.72rem",

              lineHeight:
                1.55,

              opacity:
                0.6,
            }}
          >
            {
              description
            }
          </p>
        </div>

        <span
          className={
            styles.sectionStatus
          }
        >
          {
            skipped
              ? "NON REQUIS"
              : "TERMINÉ"
          }
        </span>
      </div>
    </section>
  );
}

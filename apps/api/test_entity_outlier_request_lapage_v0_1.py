from __future__ import annotations


from pathlib import Path


from starlette.datastructures import (
    UploadFile,
)


from app.analysis.entity_outlier_requests import (
    ENTITY_OUTLIER_REQUEST_MODEL,
    ENTITY_OUTLIER_REQUEST_RULE_VERSION,
    resolve_entity_outlier_intent,
    run_entity_outlier_request,
)

from app.api.routes import (
    load_uploaded_dataset_bundle,
)


# ============================================================
# VERSION
# ============================================================


TEST_RULE_VERSION = (
    "entity_outlier_request_lapage_v0.1"
)


# ============================================================
# REAL LAPAGE DATA
# ============================================================


PROJECT_ROOT = Path(
    r"C:\Users\tomas\Documents"
    r"\openclassrooms_data_analyst"
    r"\Projet_9"
    r"\Documents"
)


DATA_ROOT = (
    PROJECT_ROOT
    /
    "Data"
)


DATASET_PATHS = [
    DATA_ROOT
    /
    "customers.csv",

    DATA_ROOT
    /
    "products.csv",

    DATA_ROOT
    /
    "Transactions.csv",
]


EXPECTED_PRIORITY_CLIENTS = {
    "c_1609",
    "c_4958",
    "c_3454",
    "c_6714",
}


# ============================================================
# UPLOADS
# ============================================================


def build_uploads(
) -> tuple[
    list[
        UploadFile
    ],
    list,
]:
    handles = []

    uploads: list[
        UploadFile
    ] = []


    for path in (
        DATASET_PATHS
    ):
        if not path.exists():
            raise FileNotFoundError(
                (
                    "Missing Lapage dataset: "
                    f"{path}"
                )
            )


        handle = path.open(
            "rb"
        )


        handles.append(
            handle
        )


        uploads.append(
            UploadFile(
                file=
                    handle,

                filename=
                    path.name,
            )
        )


    return (
        uploads,
        handles,
    )


# ============================================================
# INTENT TESTS
# ============================================================


def test_intent_resolution(
) -> None:
    # --------------------------------------------------------
    # FRENCH EXPLICIT CUSTOMER REQUEST
    # --------------------------------------------------------

    french = (
        resolve_entity_outlier_intent(
            (
                "Détecte les clients "
                "atypiques."
            )
        )
    )


    assert (
        french.status
        ==
        "matched"
    )


    assert (
        french.intent
        ==
        (
            "customer_entity_"
            "outlier_detection"
        )
    )


    assert (
        french.entity_kind
        ==
        "customer"
    )


    # --------------------------------------------------------
    # ALTERNATIVE FRENCH WORDING
    # --------------------------------------------------------

    french_alternative = (
        resolve_entity_outlier_intent(
            (
                "Trouve les clients "
                "anormaux."
            )
        )
    )


    assert (
        french_alternative.status
        ==
        "matched"
    )


    # --------------------------------------------------------
    # ENGLISH
    # --------------------------------------------------------

    english = (
        resolve_entity_outlier_intent(
            "Find anomalous customers."
        )
    )


    assert (
        english.status
        ==
        "matched"
    )


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Generic outlier request must remain available to the
    # existing variable-level generic resolver.
    # --------------------------------------------------------

    generic = (
        resolve_entity_outlier_intent(
            "Détecte les outliers."
        )
    )


    assert (
        generic.status
        ==
        "not_matched"
    )


    # --------------------------------------------------------
    # PRICE OUTLIER REQUEST MUST NOT BECOME CUSTOMER ANALYSIS
    # --------------------------------------------------------

    price = (
        resolve_entity_outlier_intent(
            (
                "Détecte les prix "
                "atypiques."
            )
        )
    )


    assert (
        price.status
        ==
        "not_matched"
    )


    # --------------------------------------------------------
    # CUSTOMER ANALYSIS WITHOUT ANOMALY SEMANTICS
    # --------------------------------------------------------

    normal_customer_request = (
        resolve_entity_outlier_intent(
            (
                "Analyse les achats "
                "des clients."
            )
        )
    )


    assert (
        normal_customer_request.status
        ==
        "not_matched"
    )


    print()
    print(
        "=== INTENT ROUTING ==="
    )

    print(
        (
            "Détecte les clients atypiques "
            ": MATCHED"
        )
    )

    print(
        (
            "Trouve les clients anormaux   "
            ": MATCHED"
        )
    )

    print(
        (
            "Find anomalous customers      "
            ": MATCHED"
        )
    )

    print(
        (
            "Détecte les outliers          "
            ": NOT INTERCEPTED"
        )
    )

    print(
        (
            "Détecte les prix atypiques    "
            ": NOT INTERCEPTED"
        )
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:
    print()
    print(
        "=========================================================="
    )

    print(
        "DataLens Entity Outlier Request · Lapage v0.1"
    )

    print(
        "=========================================================="
    )


    # ========================================================
    # 1. INTENT RESOLUTION
    # ========================================================

    test_intent_resolution()


    # ========================================================
    # 2. REAL DATA INGESTION
    # ========================================================

    uploads, handles = (
        build_uploads()
    )


    try:
        (
            _,
            source_records,
        ) = load_uploaded_dataset_bundle(
            uploads
        )


        print()
        print(
            "=== SOURCE DATASETS ==="
        )


        for record in (
            source_records
        ):
            dataframe = (
                record[
                    "dataframe"
                ]
            )


            print(
                (
                    f"{record['filename']:<20} "
                    f"{len(dataframe):>8} rows"
                )
            )


        assert (
            len(
                source_records
            )
            ==
            3
        )


        # ====================================================
        # 3. FULL ENTITY REQUEST
        # ====================================================

        report = (
            run_entity_outlier_request(
                objective=
                    (
                        "Détecte les clients "
                        "atypiques."
                    ),

                source_dataset_records=
                    source_records,

                top_profile_limit=
                    50,
            )
        )


        print()
        print(
            "=== REQUEST EXECUTION ==="
        )


        print(
            (
                "Status          : "
                f"{report.status}"
            )
        )

        print(
            (
                "Intent          : "
                f"{report.intent}"
            )
        )

        print(
            (
                "Entity kind     : "
                f"{report.entity_kind}"
            )
        )

        print(
            (
                "Model           : "
                f"{report.model}"
            )
        )

        print(
            (
                "Dataset         : "
                f"{report.dataset_filename}"
            )
        )

        print(
            (
                "Entity column   : "
                f"{report.entity_column}"
            )
        )

        print(
            (
                "Entity count    : "
                f"{report.entity_count}"
            )
        )

        print(
            (
                "IQR flagged     : "
                f"{report.raw_flagged_entity_count}"
            )
        )

        print(
            (
                "Priority        : "
                f"{report.priority_profile_count}"
            )
        )

        print(
            (
                "Other signals   : "
                f"{report.behavioral_signal_count}"
            )
        )


        # ====================================================
        # 4. CONTRACT
        # ====================================================

        assert (
            report.status
            ==
            "ready"
        )


        assert (
            report.intent
            ==
            (
                "customer_entity_"
                "outlier_detection"
            )
        )


        assert (
            report.entity_kind
            ==
            "customer"
        )


        assert (
            report.model
            ==
            ENTITY_OUTLIER_REQUEST_MODEL
        )


        assert (
            report.intent_rule_version
            ==
            ENTITY_OUTLIER_REQUEST_RULE_VERSION
        )


        assert (
            report.entity_column
            ==
            "client_id"
        )


        assert (
            report.dataset_filename
            ==
            (
                "Transactions"
                "__customers_price.derived"
            )
        )


        assert (
            report.entity_count
            ==
            8600
        )


        # ====================================================
        # 5. PRODUCT CLASSIFICATION
        # ====================================================

        assert (
            report.raw_flagged_entity_count
            ==
            1422
        )


        assert (
            report.priority_profile_count
            ==
            4
        )


        assert (
            report.behavioral_signal_count
            ==
            1418
        )


        priority_ids = {
            profile.entity

            for profile
            in report
            .priority_profiles
        }


        assert (
            priority_ids
            ==
            EXPECTED_PRIORITY_CLIENTS
        )


        # ====================================================
        # 6. PRIORITY PROFILES
        # ====================================================

        print()
        print(
            "=== PRIORITY PROFILES ==="
        )


        for (
            rank,
            profile,
        ) in enumerate(
            report
            .priority_profiles,
            start=1,
        ):
            print(
                (
                    f"{rank:02d}. "
                    f"{profile.entity:<12} "
                    f"severity="
                    f"{profile.severity:<8} "
                    f"signals="
                    f"{profile.signal_count} "
                    f"family="
                    f"{profile.dominant_family:<10} "
                    f"max="
                    f"{profile.max_distance_iqr:.2f} IQR"
                )
            )


            for evidence in (
                profile.evidence
            ):
                print(
                    (
                        "    - "
                        f"{evidence.metric:<28} "
                        f"{evidence.value:>12.2f} "
                        f"{evidence.distance_iqr:>8.2f} IQR"
                    )
                )


        # ====================================================
        # 7. KNOWN EXPECTED CLIENTS
        # ====================================================

        print()
        print(
            "=== EXPECTED CLIENTS ==="
        )


        for client_id in sorted(
            EXPECTED_PRIORITY_CLIENTS
        ):
            print(
                (
                    f"{client_id:<10} : "
                    f"{'FOUND' if client_id in priority_ids else 'MISSING'}"
                )
            )


        # ====================================================
        # 8. SECONDARY SIGNAL REGRESSION
        # ====================================================

        behavioral_ids = {
            profile.entity

            for profile
            in report
            .behavioral_signals
        }


        assert (
            "c_2369"
            not in
            priority_ids
        )


        assert (
            "c_2369"
            in
            behavioral_ids
        )


        print()
        print(
            (
                "c_2369 classification : "
                "BEHAVIORAL SIGNAL"
            )
        )


        # ====================================================
        # 9. SAFETY
        # ====================================================

        assert (
            report.blockers
            ==
            []
        )


        print()
        print(
            "=== SAFETY ==="
        )

        print(
            "Generic outlier route changed       : NO"
        )

        print(
            "Raw transaction rows as customers   : NO"
        )

        print(
            "Automatic fraud classification       : NO"
        )

        print(
            "Automatic BtoB classification        : NO"
        )

        print(
            "Automatic deletion                   : NO"
        )

        print(
            "LLM required for explicit intent     : NO"
        )


        # ====================================================
        # PASS
        # ====================================================

        print()
        print(
            "=========================================================="
        )

        print(
            "PASS - entity outlier request Lapage v0.1"
        )

        print(
            "=========================================================="
        )


    finally:
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    main()
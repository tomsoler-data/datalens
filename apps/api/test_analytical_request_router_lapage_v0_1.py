from __future__ import annotations


from pathlib import Path


from starlette.datastructures import (
    UploadFile,
)


from app.api.routes import (
    load_uploaded_dataset_bundle,
)

from app.planning.analytical_request_router import (
    ANALYTICAL_REQUEST_ROUTER_RULE_VERSION,
    route_analytical_request,
)

from app.planning.generic_intent_resolver import (
    GENERIC_ANALYTICAL_INTENT_RULE_VERSION,
)

from app.planning.planner_catalog import (
    planner_catalog_from_dataset_records,
)


# ============================================================
# VERSION
# ============================================================


TEST_RULE_VERSION = (
    "analytical_request_router_lapage_v0.1"
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
# MAIN
# ============================================================


def main(
) -> None:
    print()
    print(
        "=========================================================="
    )

    print(
        "DataLens Analytical Request Router · Lapage v0.1"
    )

    print(
        "=========================================================="
    )


    uploads, handles = (
        build_uploads()
    )


    try:
        # ====================================================
        # 1. REAL INGESTION
        # ====================================================

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
        # 2. CENTRAL PLANNER CATALOG
        # ====================================================

        catalog = (
            planner_catalog_from_dataset_records(
                source_records
            )
        )


        print()
        print(
            "=== ROUTER ==="
        )


        print(
            (
                "Rule version : "
                f"{ANALYTICAL_REQUEST_ROUTER_RULE_VERSION}"
            )
        )


        # ====================================================
        # 3. EXPLICIT CUSTOMER ENTITY REQUEST
        # ====================================================

        customer_route = (
            route_analytical_request(
                objective=
                    (
                        "Détecte les clients "
                        "atypiques."
                    ),

                source_dataset_records=
                    source_records,

                catalog=
                    catalog,

                entity_top_profile_limit=
                    50,
            )
        )


        print()
        print(
            "=== CASE 1 · CUSTOMER ENTITY OUTLIERS ==="
        )


        print(
            (
                "Objective : "
                f"{customer_route.objective}"
            )
        )

        print(
            (
                "Route     : "
                f"{customer_route.route_kind}"
            )
        )


        assert (
            customer_route
            .router_rule_version
            ==
            ANALYTICAL_REQUEST_ROUTER_RULE_VERSION
        )


        assert (
            customer_route
            .route_kind
            ==
            "entity_outlier"
        )


        assert (
            customer_route
            .planner_report
            is None
        )


        assert (
            customer_route
            .entity_outlier_report
            is not None
        )


        entity_report = (
            customer_route
            .entity_outlier_report
        )


        assert (
            entity_report.status
            ==
            "ready"
        )


        assert (
            entity_report.intent
            ==
            (
                "customer_entity_"
                "outlier_detection"
            )
        )


        assert (
            entity_report.entity_kind
            ==
            "customer"
        )


        assert (
            entity_report.entity_column
            ==
            "client_id"
        )


        assert (
            entity_report.entity_count
            ==
            8600
        )


        assert (
            entity_report
            .raw_flagged_entity_count
            ==
            1422
        )


        assert (
            entity_report
            .priority_profile_count
            ==
            4
        )


        assert (
            entity_report
            .behavioral_signal_count
            ==
            1418
        )


        priority_ids = {
            profile.entity

            for profile
            in entity_report
            .priority_profiles
        }


        assert (
            priority_ids
            ==
            EXPECTED_PRIORITY_CLIENTS
        )


        print(
            (
                "Customers : "
                f"{entity_report.entity_count}"
            )
        )

        print(
            (
                "IQR flags : "
                f"{entity_report.raw_flagged_entity_count}"
            )
        )

        print(
            (
                "Priority  : "
                f"{entity_report.priority_profile_count}"
            )
        )

        print(
            (
                "Secondary : "
                f"{entity_report.behavioral_signal_count}"
            )
        )


        print()
        print(
            "Priority profiles:"
        )


        for (
            rank,
            profile,
        ) in enumerate(
            entity_report
            .priority_profiles,
            start=1,
        ):
            print(
                (
                    f"{rank:02d}. "
                    f"{profile.entity:<10} "
                    f"{profile.dominant_family:<10} "
                    f"{profile.max_distance_iqr:.2f} IQR"
                )
            )


        # ====================================================
        # 4. GENERIC OUTLIER REQUEST
        #
        # This is the regression test that matters most.
        #
        # Adding the entity route must NOT steal:
        #
        #     "Détecte les outliers"
        #
        # from the existing generic intent router.
        # ====================================================

        generic_route = (
            route_analytical_request(
                objective=
                    "Détecte les outliers.",

                source_dataset_records=
                    source_records,

                catalog=
                    catalog,
            )
        )


        print()
        print(
            "=== CASE 2 · GENERIC VARIABLE OUTLIERS ==="
        )


        print(
            (
                "Objective : "
                f"{generic_route.objective}"
            )
        )

        print(
            (
                "Route     : "
                f"{generic_route.route_kind}"
            )
        )


        assert (
            generic_route
            .route_kind
            ==
            "analytical_plan"
        )


        assert (
            generic_route
            .entity_outlier_report
            is None
        )


        assert (
            generic_route
            .planner_report
            is not None
        )


        planner_report = (
            generic_route
            .planner_report
        )


        expected_model = (
            "python:"
            f"{GENERIC_ANALYTICAL_INTENT_RULE_VERSION}"
        )


        assert (
            planner_report.model
            ==
            expected_model
        )


        assert (
            planner_report
            .timing
            .model_inference_ms
            ==
            0.0
        )


        assert (
            planner_report
            .validated_count
            >=
            1
        )


        validated_items = [
            item

            for item
            in planner_report.items

            if (
                item.validation_status
                ==
                "validated"
            )
        ]


        assert (
            validated_items
        )


        validated_value_columns = {
            item.proposal.value_column

            for item
            in validated_items

            if (
                item.proposal.value_column
                is not None
            )
        }


        print(
            (
                "Planner   : "
                f"{planner_report.model}"
            )
        )

        print(
            (
                "Inference : "
                f"{planner_report.timing.model_inference_ms:.2f} ms"
            )
        )

        print(
            (
                "Validated : "
                f"{planner_report.validated_count}"
            )
        )

        print(
            (
                "Targets   : "
                f"{sorted(validated_value_columns)}"
            )
        )


        # ====================================================
        # 5. LAPAGE TYPING REGRESSION
        #
        # We already validated centrally:
        #
        # birth     -> temporal
        # categ     -> categorical
        # client_id -> identifier
        # price     -> quantitative
        #
        # Therefore the generic raw-variable outlier request
        # should still resolve to price.
        # ====================================================

        assert (
            "price"
            in
            validated_value_columns
        )


        assert (
            "birth"
            not in
            validated_value_columns
        )


        assert (
            "categ"
            not in
            validated_value_columns
        )


        assert (
            "client_id"
            not in
            validated_value_columns
        )


        # ====================================================
        # 6. ROUTE SEPARATION
        # ====================================================

        print()
        print(
            "=== ROUTE SEPARATION ==="
        )


        print(
            (
                "Détecte les clients atypiques "
                "→ entity_outlier"
            )
        )

        print(
            (
                "Détecte les outliers         "
                "→ analytical_plan"
            )
        )


        assert (
            customer_route.route_kind
            !=
            generic_route.route_kind
        )


        # ====================================================
        # 7. SAFETY
        # ====================================================

        print()
        print(
            "=== SAFETY ==="
        )

        print(
            "Explicit entity intent has priority : YES"
        )

        print(
            "Generic outlier regression preserved: YES"
        )

        print(
            "Customer rows derived safely         : YES"
        )

        print(
            "LLM used for generic outliers        : NO"
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


        # ====================================================
        # PASS
        # ====================================================

        print()
        print(
            "=========================================================="
        )

        print(
            "PASS - analytical request router Lapage v0.1"
        )

        print(
            "=========================================================="
        )


    finally:
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    main()
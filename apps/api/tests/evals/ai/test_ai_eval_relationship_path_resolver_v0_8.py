from __future__ import annotations

from app.evals.relationship_path_resolver_v0_8 import (
    RELATIONSHIP_PATH_RESOLVER_VERSION,
    resolve_validated_relationship_plan,
)

from app.evals.routing_relationships_v0_8 import (
    RoutingRelationshipContext,
)


# ============================================================
# CONTEXT
# ============================================================

def make_context() -> RoutingRelationshipContext:

    return (
        RoutingRelationshipContext
        .model_validate(
            {
                "datasets": [

                    # ========================================
                    # PATIENTS — bridge dataset
                    # ========================================

                    {
                        "dataset_id":
                            "patients",

                        "filename":
                            "patients.csv",

                        "grain":
                            "patient",

                        "entity_columns": [
                            "patient_id",
                        ],

                        "columns": [
                            {
                                "name":
                                    "patient_id",

                                "analytical_type":
                                    "identifier",
                            },
                        ],
                    },


                    # ========================================
                    # CONSULTATIONS
                    # ========================================

                    {
                        "dataset_id":
                            "consultations",

                        "filename":
                            "consultations.csv",

                        "grain":
                            "patient_month",

                        "entity_columns": [
                            "patient_id",
                        ],

                        "columns": [
                            {
                                "name":
                                    "patient_id",

                                "analytical_type":
                                    "identifier",
                            },

                            {
                                "name":
                                    "consultation_count",

                                "analytical_type":
                                    "quantitative",
                            },
                        ],
                    },


                    # ========================================
                    # CARE COSTS
                    # ========================================

                    {
                        "dataset_id":
                            "care_costs",

                        "filename":
                            "care_costs.csv",

                        "grain":
                            "patient_month",

                        "entity_columns": [
                            "patient_id",
                        ],

                        "columns": [
                            {
                                "name":
                                    "patient_id",

                                "analytical_type":
                                    "identifier",
                            },

                            {
                                "name":
                                    "total_care_cost",

                                "analytical_type":
                                    "quantitative",
                            },
                        ],
                    },


                    # ========================================
                    # CLAIMS — disconnected
                    # ========================================

                    {
                        "dataset_id":
                            "claims",

                        "filename":
                            "claims.csv",

                        "grain":
                            "claim",

                        "entity_columns": [
                            "claim_id",
                        ],

                        "columns": [
                            {
                                "name":
                                    "claim_id",

                                "analytical_type":
                                    "identifier",
                            },

                            {
                                "name":
                                    "claim_amount",

                                "analytical_type":
                                    "quantitative",
                            },
                        ],
                    },
                ],

                "relationships": [

                    {
                        "relationship_id":
                            "patients_consultations",

                        "left_dataset_id":
                            "patients",

                        "right_dataset_id":
                            "consultations",

                        "kind":
                            "join",

                        "left_keys": [
                            "patient_id",
                        ],

                        "right_keys": [
                            "patient_id",
                        ],

                        "validated":
                            True,
                    },


                    {
                        "relationship_id":
                            "patients_costs",

                        "left_dataset_id":
                            "patients",

                        "right_dataset_id":
                            "care_costs",

                        "kind":
                            "join",

                        "left_keys": [
                            "patient_id",
                        ],

                        "right_keys": [
                            "patient_id",
                        ],

                        "validated":
                            True,
                    },
                ],

                "available_tools": [
                    "aggregate",
                    "measure_association",
                    "join_datasets",
                ],
            }
        )
    )


# ============================================================
# 1. SINGLE DATASET
# ============================================================

def test_single_dataset() -> None:

    result = (
        resolve_validated_relationship_plan(
            context=(
                make_context()
            ),

            required_dataset_ids=[
                "consultations",
            ],
        )
    )


    assert result.connected


    assert (
        result.required_dataset_ids
        == [
            "consultations",
        ]
    )


    assert (
        result.bridge_dataset_ids
        == []
    )


    assert (
        result.relationship_ids
        == []
    )


    assert (
        result.paths
        == []
    )


    print(
        "Single dataset resolution: PASS"
    )


# ============================================================
# 2. DIRECT PATH
# ============================================================

def test_direct_path() -> None:

    result = (
        resolve_validated_relationship_plan(
            context=(
                make_context()
            ),

            required_dataset_ids=[
                "patients",
                "consultations",
            ],
        )
    )


    assert result.connected


    assert (
        result.bridge_dataset_ids
        == []
    )


    assert (
        result.relationship_ids
        == [
            "patients_consultations",
        ]
    )


    assert (
        len(
            result.paths
        )
        == 1
    )


    path = (
        result.paths[
            0
        ]
    )


    assert (
        path.dataset_ids
        == [
            "patients",
            "consultations",
        ]
    )


    assert (
        len(
            path.steps
        )
        == 1
    )


    print(
        "Direct relationship path: PASS"
    )


# ============================================================
# 3. REVERSED TRAVERSAL
# ============================================================

def test_reversed_traversal_keys() -> None:

    result = (
        resolve_validated_relationship_plan(
            context=(
                make_context()
            ),

            required_dataset_ids=[
                "consultations",
                "patients",
            ],
        )
    )


    assert result.connected


    step = (
        result
        .paths[
            0
        ]
        .steps[
            0
        ]
    )


    assert (
        step.from_dataset_id
        == "consultations"
    )


    assert (
        step.to_dataset_id
        == "patients"
    )


    assert (
        step.from_keys
        == [
            "patient_id",
        ]
    )


    assert (
        step.to_keys
        == [
            "patient_id",
        ]
    )


    print(
        "Reversed traversal orientation: PASS"
    )


# ============================================================
# 4. MULTI-HOP BRIDGE
# ============================================================

def test_multihop_bridge_dataset() -> None:

    result = (
        resolve_validated_relationship_plan(
            context=(
                make_context()
            ),

            required_dataset_ids=[
                "consultations",
                "care_costs",
            ],
        )
    )


    assert result.connected


    # Semantic dependencies remain unchanged.
    assert (
        result.required_dataset_ids
        == [
            "consultations",
            "care_costs",
        ]
    )


    # Python discovers the structural bridge.
    assert (
        result.bridge_dataset_ids
        == [
            "patients",
        ]
    )


    assert (
        result.all_dataset_ids
        == [
            "consultations",
            "care_costs",
            "patients",
        ]
    )


    assert (
        result.relationship_ids
        == [
            "patients_consultations",
            "patients_costs",
        ]
    )


    path = (
        result.paths[
            0
        ]
    )


    assert (
        path.dataset_ids
        == [
            "consultations",
            "patients",
            "care_costs",
        ]
    )


    assert (
        path.relationship_ids
        == [
            "patients_consultations",
            "patients_costs",
        ]
    )


    print(
        "Multi-hop bridge resolution: PASS"
    )


# ============================================================
# 5. THREE REQUIRED DATASETS
# ============================================================

def test_three_required_datasets() -> None:

    result = (
        resolve_validated_relationship_plan(
            context=(
                make_context()
            ),

            required_dataset_ids=[
                "patients",
                "consultations",
                "care_costs",
            ],
        )
    )


    assert result.connected


    assert (
        result.bridge_dataset_ids
        == []
    )


    assert (
        set(
            result.relationship_ids
        )
        == {
            "patients_consultations",
            "patients_costs",
        }
    )


    assert (
        len(
            result.paths
        )
        == 2
    )


    print(
        "Three required datasets: PASS"
    )


# ============================================================
# 6. DISCONNECTED DATASET
# ============================================================

def test_disconnected_dataset() -> None:

    result = (
        resolve_validated_relationship_plan(
            context=(
                make_context()
            ),

            required_dataset_ids=[
                "consultations",
                "claims",
            ],
        )
    )


    assert not (
        result.connected
    )


    assert (
        result.unresolved_dataset_ids
        == [
            "claims",
        ]
    )


    assert (
        result.relationship_ids
        == []
    )


    print(
        "Disconnected dataset detected: PASS"
    )


# ============================================================
# 7. MIXED CONNECTED + DISCONNECTED
# ============================================================

def test_partial_resolution() -> None:

    result = (
        resolve_validated_relationship_plan(
            context=(
                make_context()
            ),

            required_dataset_ids=[
                "consultations",
                "care_costs",
                "claims",
            ],
        )
    )


    assert not (
        result.connected
    )


    # consultations -> patients -> care_costs is valid.
    assert (
        len(
            result.paths
        )
        == 1
    )


    assert (
        result.bridge_dataset_ids
        == [
            "patients",
        ]
    )


    # claims cannot be reached.
    assert (
        result.unresolved_dataset_ids
        == [
            "claims",
        ]
    )


    print(
        "Partial relationship resolution: PASS"
    )


# ============================================================
# 8. DUPLICATE REQUIRED DATASET
# ============================================================

def test_duplicate_required_dataset() -> None:

    result = (
        resolve_validated_relationship_plan(
            context=(
                make_context()
            ),

            required_dataset_ids=[
                "consultations",
                "consultations",
                "care_costs",
            ],
        )
    )


    assert (
        result.required_dataset_ids
        == [
            "consultations",
            "care_costs",
        ]
    )


    assert result.connected


    print(
        "Duplicate required dataset normalized: PASS"
    )


# ============================================================
# 9. UNKNOWN REQUIRED DATASET
# ============================================================

def test_unknown_dataset_rejected() -> None:

    try:

        resolve_validated_relationship_plan(
            context=(
                make_context()
            ),

            required_dataset_ids=[
                "consultations",
                "invented_dataset",
            ],
        )


    except ValueError as error:

        assert (
            "invented_dataset"
            in str(
                error
            )
        )


        print(
            "Unknown required dataset rejected: PASS"
        )


    else:

        raise AssertionError(
            "Unknown required datasets must be rejected."
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=== DATALENS RELATIONSHIP PATH RESOLVER v0.8 ==="
    )


    print(
        "Resolver:",
        RELATIONSHIP_PATH_RESOLVER_VERSION,
    )


    print()


    test_single_dataset()

    test_direct_path()

    test_reversed_traversal_keys()

    test_multihop_bridge_dataset()

    test_three_required_datasets()

    test_disconnected_dataset()

    test_partial_resolution()

    test_duplicate_required_dataset()

    test_unknown_dataset_rejected()


    print()

    print(
        "Relationship Path Resolver v0.8: PASS"
    )


if __name__ == "__main__":
    main()
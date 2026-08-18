from __future__ import annotations


from collections import (
    deque,
)

from dataclasses import (
    dataclass,
)

from hashlib import (
    sha256,
)

import re

import unicodedata

from typing import (
    TYPE_CHECKING,
)


from app.ingestion.schemas import (
    DatasetColumnManifest,
    DatasetManifest,
    MultiDatasetIngestion,
)

from app.planning.schemas import (
    RequestedAnalysisKind,
    RequestedAnalysisPlan,
    RequestedAnalysisPlanReport,
    RequestedColumnMatch,
)


if TYPE_CHECKING:
    from app.document_summary import (
        VerifiedDocumentClaim,
    )


# ============================================================
# VERSION
# ============================================================

REQUEST_PLANNER_RULE_VERSION = (
    "analytical_request_planner_v0.2"
)


# ============================================================
# INTERNAL COLUMN REFERENCE
# ============================================================

@dataclass(
    frozen=True
)
class ColumnReference:
    dataset_id: str

    dataset_filename: str

    column: DatasetColumnManifest

    concept: str

    score: int

    reasons: tuple[
        str,
        ...
    ]


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(
    value: str,
) -> str:
    value = (
        unicodedata.normalize(
            "NFKC",
            value,
        )
    )


    value = (
        unicodedata.normalize(
            "NFKD",
            value,
        )
        .encode(
            "ascii",
            "ignore",
        )
        .decode(
            "ascii",
        )
        .casefold()
    )


    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value,
    )


    return (
        " ".join(
            value.split()
        )
    )


def compact_text(
    value: str,
) -> str:
    return (
        normalize_text(
            value
        )
        .replace(
            " ",
            "",
        )
    )


def text_tokens(
    value: str,
) -> set[
    str
]:
    return set(
        normalize_text(
            value
        ).split()
    )


# ============================================================
# COLUMN CONCEPT ALIASES
# ============================================================

COLUMN_ALIASES = {
    "customer_id": {
        "client id",
        "client_id",
        "customer id",
        "customer_id",
        "id client",
        "id_client",
        "id customer",
        "id_customer",
    },

    "product_id": {
        "product id",
        "product_id",
        "id product",
        "id_product",
        "produit id",
        "produit_id",
        "id produit",
        "id_produit",
        "prod id",
        "prod_id",
        "id prod",
        "id_prod",
    },

    "session_id": {
        "session id",
        "session_id",
        "id session",
        "id_session",
        "basket id",
        "basket_id",
        "cart id",
        "cart_id",
        "order id",
        "order_id",
        "commande id",
        "commande_id",
    },

    "transaction_id": {
        "transaction id",
        "transaction_id",
        "id transaction",
        "id_transaction",
        "order id",
        "order_id",
        "commande id",
        "commande_id",
    },

    "gender": {
        "sex",
        "sexe",
        "gender",
        "genre",
    },

    "category": {
        "category",
        "categorie",
        "categ",
        "product category",
        "product_category",
        "categorie produit",
        "categorie_produit",
    },

    "age": {
        "age",
        "customer age",
        "customer_age",
        "client age",
        "client_age",
        "age at first purchase",
        "age_at_first_purchase",
        "age premier achat",
        "age_premier_achat",
    },

    "birth": {
        "birth",
        "birth year",
        "birth_year",
        "birth date",
        "birth_date",
        "date birth",
        "date_birth",
        "annee naissance",
        "annee_naissance",
        "date naissance",
        "date_naissance",
        "naissance",
    },

    "time": {
        "date",
        "datetime",
        "timestamp",
        "transaction date",
        "transaction_date",
        "purchase date",
        "purchase_date",
        "order date",
        "order_date",
        "date achat",
        "date_achat",
        "date transaction",
        "date_transaction",
        "year",
        "annee",
    },

    "amount": {
        "price",
        "prix",
        "amount",
        "montant",
        "value",
        "revenue",
        "sales",
        "turnover",
        "chiffre affaires",
        "chiffre_affaires",
        "ca",
    },

    "b2b_explicit": {
        "b2b",
        "btob",
        "is b2b",
        "is_b2b",
        "b2b customer",
        "b2b_customer",
        "business customer",
        "business_customer",
    },

    "customer_type": {
        "customer type",
        "customer_type",
        "client type",
        "client_type",
        "customer segment",
        "customer_segment",
        "client segment",
        "client_segment",
        "segment",
    },
}


# ============================================================
# REQUEST TEXT CLASSIFICATION
# ============================================================

def request_search_text(
    claim: VerifiedDocumentClaim,
) -> str:
    values = [
        claim.statement,
        claim.evidence_quote,
    ]


    if claim.context_quote:
        values.append(
            claim.context_quote
        )


    return normalize_text(
        " ".join(
            values
        )
    )


def classify_request(
    claim: VerifiedDocumentClaim,
) -> RequestedAnalysisKind:
    text = (
        request_search_text(
            claim
        )
    )


    # --------------------------------------------------------
    # BUSINESS SEGMENT
    # --------------------------------------------------------

    if (
        "b2b"
        in text
        or
        "btob"
        in text
    ):
        return (
            "b2b_revenue_distribution"
        )


    # --------------------------------------------------------
    # INEQUALITY
    # --------------------------------------------------------

    if (
        "lorenz"
        in text
    ):
        return (
            "lorenz_curve"
        )


    # --------------------------------------------------------
    # REVENUE
    # --------------------------------------------------------

    if (
        "chiffre d affaires"
        in text
        and
        "moyenne mobile"
        in text
    ):
        return (
            "revenue_moving_average"
        )


    if (
        "chiffre d affaires"
        in text
        and
        "categorie"
        in text
    ):
        return (
            "revenue_by_category"
        )


    # --------------------------------------------------------
    # COUNTS
    # --------------------------------------------------------

    if (
        "nombre de clients"
        in text
        and
        (
            "mois"
            in text
            or
            "month"
            in text
        )
    ):
        return (
            "customers_by_period"
        )


    if (
        "nombre de transactions"
        in text
    ):
        return (
            "transaction_count"
        )


    if (
        "nombre de produits vendus"
        in text
    ):
        return (
            "products_sold_count"
        )


    # --------------------------------------------------------
    # PRODUCT REFERENCES
    # --------------------------------------------------------

    if (
        "tops"
        in text
        or
        re.search(
            r"\btop\b",
            text,
        )
    ):
        return (
            "top_products"
        )


    if (
        "flops"
        in text
        or
        re.search(
            r"\bflop\b",
            text,
        )
    ):
        return (
            "flop_products"
        )


    if (
        "repartition"
        in text
        and
        "categorie"
        in text
        and
        (
            "reference"
            in text
            or
            "references"
            in text
        )
    ):
        return (
            "product_category_distribution"
        )


    # --------------------------------------------------------
    # RELATIONSHIPS
    # --------------------------------------------------------

    has_age = (
        "age"
        in text_tokens(
            text
        )
    )


    has_category = (
        "categorie"
        in text
        or
        "categories"
        in text
    )


    has_gender = (
        "genre"
        in text
        or
        "sexe"
        in text
        or
        "gender"
        in text
    )


    has_total_amount = (
        "montant total"
        in text
        or
        "total des achats"
        in text
        or
        "total achats"
        in text
    )


    has_frequency = (
        "frequence"
        in text
    )


    has_average_basket = (
        "panier moyen"
        in text
    )


    if (
        has_gender
        and
        has_category
    ):
        return (
            "gender_category_association"
        )


    if (
        has_age
        and
        has_total_amount
    ):
        return (
            "age_total_amount_association"
        )


    if (
        has_age
        and
        has_frequency
    ):
        return (
            "age_frequency_association"
        )


    if (
        has_age
        and
        has_average_basket
    ):
        return (
            "age_average_basket_association"
        )


    if (
        has_age
        and
        has_category
    ):
        return (
            "age_category_association"
        )


    return "unknown"


# ============================================================
# COLUMN MATCHING
# ============================================================

def score_column_for_concept(
    column_name: str,
    concept: str,
) -> tuple[
    int,
    list[
        str
    ],
]:
    aliases = (
        COLUMN_ALIASES.get(
            concept,
            set(),
        )
    )


    normalized_name = (
        normalize_text(
            column_name
        )
    )


    compact_name = (
        compact_text(
            column_name
        )
    )


    name_tokens = (
        text_tokens(
            column_name
        )
    )


    # Prevent a birth column from being selected as
    # transaction time.
    if (
        concept
        ==
        "time"
        and
        (
            "birth"
            in normalized_name
            or
            "naissance"
            in normalized_name
        )
    ):
        return (
            0,
            [],
        )


    best_score = 0

    reasons: list[
        str
    ] = []


    for alias in aliases:
        normalized_alias = (
            normalize_text(
                alias
            )
        )


        compact_alias = (
            compact_text(
                alias
            )
        )


        alias_tokens = (
            text_tokens(
                alias
            )
        )


        score = 0


        if (
            normalized_name
            ==
            normalized_alias
        ):
            score = 100


        elif (
            compact_name
            ==
            compact_alias
        ):
            score = 96


        elif (
            alias_tokens
            and
            alias_tokens.issubset(
                name_tokens
            )
        ):
            score = 90


        elif (
            normalized_alias
            and
            normalized_alias
            in
            normalized_name
        ):
            score = 82


        elif (
            alias_tokens
            and
            name_tokens
        ):
            overlap = (
                alias_tokens
                &
                name_tokens
            )


            if overlap:
                score = (
                    55
                    +
                    min(
                        len(
                            overlap
                        )
                        *
                        8,
                        24,
                    )
                )


        if (
            score
            >
            best_score
        ):
            best_score = (
                score
            )


            reasons = [
                (
                    f"La colonne {column_name} "
                    f"correspond au concept "
                    f"{concept} via l'alias "
                    f"{alias!r}."
                )
            ]


    return (
        best_score,
        reasons,
    )


def find_column_matches(
    manifests: list[
        DatasetManifest
    ],
    concept: str,
    *,
    limit: int = 8,
    preferred_dataset_ids: set[
        str
    ] | None = None,
) -> list[
    ColumnReference
]:
    matches: list[
        ColumnReference
    ] = []


    preferred_dataset_ids = (
        preferred_dataset_ids
        or
        set()
    )


    for manifest in manifests:
        for column in manifest.columns:
            (
                score,
                reasons,
            ) = score_column_for_concept(
                column.name,
                concept,
            )


            if (
                score
                <
                70
            ):
                continue


            if (
                manifest.dataset_id
                in
                preferred_dataset_ids
            ):
                score = min(
                    score
                    +
                    8,
                    100,
                )


                reasons = [
                    *reasons,
                    (
                        "Ce dataset est prioritaire "
                        "pour cette opération."
                    ),
                ]


            matches.append(
                ColumnReference(
                    dataset_id=
                        manifest.dataset_id,

                    dataset_filename=
                        manifest.filename,

                    column=
                        column,

                    concept=
                        concept,

                    score=
                        score,

                    reasons=
                        tuple(
                            reasons
                        ),
                )
            )


    matches.sort(
        key=lambda match: (
            match.score,
            -match.column.missing_ratio,
            match.dataset_filename.casefold(),
            match.column.name.casefold(),
        ),
        reverse=True,
    )


    return matches[
        :
        limit
    ]


def best_column_match(
    manifests: list[
        DatasetManifest
    ],
    concept: str,
    *,
    preferred_dataset_ids: set[
        str
    ] | None = None,
) -> (
    ColumnReference
    | None
):
    matches = (
        find_column_matches(
            manifests,
            concept,
            limit=1,
            preferred_dataset_ids=
                preferred_dataset_ids,
        )
    )


    if not matches:
        return None


    return matches[
        0
    ]


# ============================================================
# TRANSACTION DATASET DETECTION
# ============================================================

TRANSACTION_FILENAME_TERMS = {
    "transaction",
    "transactions",
    "order",
    "orders",
    "commande",
    "commandes",
    "purchase",
    "purchases",
    "achat",
    "achats",
    "sale",
    "sales",
    "vente",
    "ventes",
}


def manifest_has_concept(
    manifest: DatasetManifest,
    concept: str,
) -> bool:
    return any(
        score_column_for_concept(
            column.name,
            concept,
        )[0]
        >=
        70

        for column
        in manifest.columns
    )


def transaction_dataset_score(
    manifest: DatasetManifest,
) -> int:
    filename_tokens = (
        text_tokens(
            manifest.filename
        )
    )


    score = 0


    if (
        filename_tokens
        &
        TRANSACTION_FILENAME_TERMS
    ):
        score += 70


    if manifest_has_concept(
        manifest,
        "customer_id",
    ):
        score += 12


    if manifest_has_concept(
        manifest,
        "product_id",
    ):
        score += 12


    if manifest_has_concept(
        manifest,
        "session_id",
    ):
        score += 10


    if manifest_has_concept(
        manifest,
        "time",
    ):
        score += 10


    return score


def find_transaction_dataset(
    manifests: list[
        DatasetManifest
    ],
) -> (
    DatasetManifest
    | None
):
    scored = [
        (
            transaction_dataset_score(
                manifest
            ),
            manifest,
        )

        for manifest
        in manifests
    ]


    scored = [
        item

        for item
        in scored

        if (
            item[
                0
            ]
            >=
            20
        )
    ]


    if not scored:
        return None


    scored.sort(
        key=lambda item: (
            item[
                0
            ],
            item[
                1
            ].row_count,
        ),
        reverse=True,
    )


    return scored[
        0
    ][
        1
    ]


# ============================================================
# DATASET RELATIONSHIPS
# ============================================================

def identifier_concept(
    column_name: str,
) -> (
    str
    | None
):
    for concept in (
        "customer_id",
        "product_id",
        "session_id",
        "transaction_id",
    ):
        score, _ = (
            score_column_for_concept(
                column_name,
                concept,
            )
        )


        if (
            score
            >=
            80
        ):
            return concept


    return None


def manifest_identifier_concepts(
    manifest: DatasetManifest,
) -> set[
    str
]:
    concepts: set[
        str
    ] = set()


    for column in manifest.columns:
        concept = (
            identifier_concept(
                column.name
            )
        )


        if concept:
            concepts.add(
                concept
            )


    return concepts


def build_dataset_graph(
    manifests: list[
        DatasetManifest
    ],
) -> dict[
    str,
    set[
        str
    ],
]:
    graph = {
        manifest.dataset_id:
            set()

        for manifest
        in manifests
    }


    identifiers = {
        manifest.dataset_id:
            manifest_identifier_concepts(
                manifest
            )

        for manifest
        in manifests
    }


    for left_index in range(
        len(
            manifests
        )
    ):
        left = manifests[
            left_index
        ]


        for right_index in range(
            left_index + 1,
            len(
                manifests
            ),
        ):
            right = manifests[
                right_index
            ]


            shared_identifiers = (
                identifiers[
                    left.dataset_id
                ]
                &
                identifiers[
                    right.dataset_id
                ]
            )


            if not shared_identifiers:
                continue


            graph[
                left.dataset_id
            ].add(
                right.dataset_id
            )


            graph[
                right.dataset_id
            ].add(
                left.dataset_id
            )


    return graph


def shortest_dataset_path(
    graph: dict[
        str,
        set[
            str
        ]
    ],
    start: str,
    target: str,
) -> (
    list[
        str
    ]
    | None
):
    if (
        start
        ==
        target
    ):
        return [
            start
        ]


    queue = deque(
        [
            (
                start,
                [
                    start
                ],
            )
        ]
    )


    visited = {
        start
    }


    while queue:
        (
            current,
            path,
        ) = queue.popleft()


        for neighbour in (
            graph.get(
                current,
                set(),
            )
        ):
            if (
                neighbour
                in
                visited
            ):
                continue


            next_path = [
                *path,
                neighbour,
            ]


            if (
                neighbour
                ==
                target
            ):
                return next_path


            visited.add(
                neighbour
            )


            queue.append(
                (
                    neighbour,
                    next_path,
                )
            )


    return None


def connect_required_datasets(
    manifests: list[
        DatasetManifest
    ],
    required_dataset_ids: set[
        str
    ],
) -> (
    list[
        str
    ]
    | None
):
    if not required_dataset_ids:
        return []


    if (
        len(
            required_dataset_ids
        )
        ==
        1
    ):
        return list(
            required_dataset_ids
        )


    graph = (
        build_dataset_graph(
            manifests
        )
    )


    ordered_targets = list(
        required_dataset_ids
    )


    connected = {
        ordered_targets[
            0
        ]
    }


    for target in (
        ordered_targets[
            1:
        ]
    ):
        if (
            target
            in
            connected
        ):
            continue


        best_path: (
            list[
                str
            ]
            | None
        ) = None


        for connected_dataset in list(
            connected
        ):
            path = (
                shortest_dataset_path(
                    graph,
                    connected_dataset,
                    target,
                )
            )


            if path is None:
                continue


            if (
                best_path is None
                or
                len(
                    path
                )
                <
                len(
                    best_path
                )
            ):
                best_path = (
                    path
                )


        if best_path is None:
            return None


        connected.update(
            best_path
        )


    manifest_order = {
        manifest.dataset_id:
            index

        for (
            index,
            manifest,
        ) in enumerate(
            manifests
        )
    }


    return sorted(
        connected,
        key=lambda dataset_id:
            manifest_order.get(
                dataset_id,
                999999,
            ),
    )


# ============================================================
# MATCH CONVERSION
# ============================================================

def public_match(
    reference: ColumnReference,
) -> RequestedColumnMatch:
    return (
        RequestedColumnMatch(
            concept=
                reference.concept,

            dataset_id=
                reference.dataset_id,

            dataset_filename=
                reference.dataset_filename,

            column=
                reference.column.name,

            analysis_kind=
                reference.column.analysis_kind,

            match_score=
                reference.score,

            reasons=
                list(
                    reference.reasons
                ),
        )
    )


def deduplicate_matches(
    matches: list[
        ColumnReference
    ],
) -> list[
    ColumnReference
]:
    seen: set[
        tuple[
            str,
            str,
            str,
        ]
    ] = set()


    output: list[
        ColumnReference
    ] = []


    for match in matches:
        key = (
            match.concept,
            match.dataset_id,
            match.column.name,
        )


        if key in seen:
            continue


        seen.add(
            key
        )


        output.append(
            match
        )


    return output


# ============================================================
# AGE RESOLUTION
# ============================================================

def resolve_age(
    manifests: list[
        DatasetManifest
    ],
    transaction_dataset: (
        DatasetManifest
        | None
    ),
) -> tuple[
    list[
        ColumnReference
    ],
    list[
        str
    ],
    list[
        str
    ],
]:
    direct_age = (
        best_column_match(
            manifests,
            "age",
        )
    )


    if direct_age:
        return (
            [
                direct_age
            ],
            [
                (
                    "Utiliser la variable d'âge "
                    "explicitement disponible."
                )
            ],
            [],
        )


    birth = (
        best_column_match(
            manifests,
            "birth",
        )
    )


    preferred_time_datasets = (
        {
            transaction_dataset.dataset_id
        }

        if transaction_dataset
        else
        set()
    )


    transaction_time = (
        best_column_match(
            manifests,
            "time",
            preferred_dataset_ids=
                preferred_time_datasets,
        )
    )


    customer_id = (
        best_column_match(
            manifests,
            "customer_id",
            preferred_dataset_ids=
                preferred_time_datasets,
        )
    )


    blockers: list[
        str
    ] = []


    if birth is None:
        blockers.append(
            (
                "Aucune variable d'âge ni "
                "information de naissance "
                "fiable n'a été identifiée."
            )
        )


    if transaction_time is None:
        blockers.append(
            (
                "Aucune date d'achat ou variable "
                "temporelle exploitable n'a été "
                "identifiée pour dériver l'âge "
                "au moment de l'achat."
            )
        )


    if customer_id is None:
        blockers.append(
            (
                "Aucun identifiant client fiable "
                "n'a été identifié pour relier "
                "la naissance aux achats."
            )
        )


    matches = [
        match

        for match
        in [
            birth,
            transaction_time,
            customer_id,
        ]

        if match is not None
    ]


    return (
        matches,
        [
            (
                "Relier les informations client "
                "aux transactions."
            ),
            (
                "Déterminer la date du premier "
                "achat par client."
            ),
            (
                "Calculer l'âge du client au "
                "premier achat de manière "
                "déterministe."
            ),
        ],
        blockers,
    )


# ============================================================
# PLAN BUILDING HELPERS
# ============================================================

def request_identifier(
    claim: VerifiedDocumentClaim,
) -> str:
    raw_key = (
        f"{claim.citation.chunk_id}:"
        f"{claim.evidence_unit_id}"
    )


    digest = (
        sha256(
            raw_key.encode(
                "utf-8"
            )
        )
        .hexdigest()[
            :16
        ]
    )


    return (
        f"request:{digest}"
    )


def dataset_filenames_for_ids(
    manifests: list[
        DatasetManifest
    ],
    dataset_ids: list[
        str
    ],
) -> list[
    str
]:
    lookup = {
        manifest.dataset_id:
            manifest.filename

        for manifest
        in manifests
    }


    return [
        lookup[
            dataset_id
        ]

        for dataset_id
        in dataset_ids

        if dataset_id
        in lookup
    ]


def base_plan_values(
    claim: VerifiedDocumentClaim,
) -> dict:
    return {
        "request_id":
            request_identifier(
                claim
            ),

        "request_text":
            claim.statement,

        "context_text":
            claim.context_quote,

        "evidence_quote":
            claim.evidence_quote,

        "source_filename":
            claim.citation.filename,

        "source_locator":
            claim.citation.source_locator,

        "page_number":
            claim.citation.page_number,

        "source_chunk_id":
            claim.citation.chunk_id,

        "evidence_unit_id":
            claim.evidence_unit_id,
    }


def finalize_plan(
    *,
    claim: VerifiedDocumentClaim,
    kind: RequestedAnalysisKind,
    target_family: str,
    manifests: list[
        DatasetManifest
    ],
    matches: list[
        ColumnReference
    ],
    required_dataset_ids: set[
        str
    ],
    operations: list[
        str
    ],
    reasons: list[
        str
    ],
    blockers: list[
        str
    ],
) -> RequestedAnalysisPlan:
    matches = (
        deduplicate_matches(
            matches
        )
    )


    if blockers:
        dataset_ids = list(
            dict.fromkeys(
                [
                    *required_dataset_ids,
                    *[
                        match.dataset_id
                        for match
                        in matches
                    ],
                ]
            )
        )


        return (
            RequestedAnalysisPlan(
                **base_plan_values(
                    claim
                ),

                kind=
                    kind,

                status=
                    "blocked",

                target_family=
                    target_family,

                matched_columns=[
                    public_match(
                        match
                    )

                    for match
                    in matches
                ],

                required_dataset_ids=
                    dataset_ids,

                required_dataset_filenames=
                    dataset_filenames_for_ids(
                        manifests,
                        dataset_ids,
                    ),

                required_operations=
                    operations,

                reasons=
                    reasons,

                blockers=
                    blockers,
            )
        )


    dataset_ids = {
        *required_dataset_ids,
        *[
            match.dataset_id
            for match
            in matches
        ],
    }


    connected_dataset_ids = (
        connect_required_datasets(
            manifests,
            dataset_ids,
        )
    )


    if (
        connected_dataset_ids
        is None
    ):
        sorted_dataset_ids = sorted(
            dataset_ids
        )


        return (
            RequestedAnalysisPlan(
                **base_plan_values(
                    claim
                ),

                kind=
                    kind,

                status=
                    "blocked",

                target_family=
                    target_family,

                matched_columns=[
                    public_match(
                        match
                    )

                    for match
                    in matches
                ],

                required_dataset_ids=
                    sorted_dataset_ids,

                required_dataset_filenames=
                    dataset_filenames_for_ids(
                        manifests,
                        sorted_dataset_ids,
                    ),

                required_operations=
                    operations,

                reasons=
                    reasons,

                blockers=[
                    (
                        "Les variables nécessaires "
                        "ont été identifiées, mais "
                        "aucun chemin de jointure "
                        "fiable par identifiants "
                        "n'a été trouvé entre les "
                        "datasets concernés."
                    )
                ],
            )
        )


    return (
        RequestedAnalysisPlan(
            **base_plan_values(
                claim
            ),

            kind=
                kind,

            status=
                "ready",

            target_family=
                target_family,

            matched_columns=[
                public_match(
                    match
                )

                for match
                in matches
            ],

            required_dataset_ids=
                connected_dataset_ids,

            required_dataset_filenames=
                dataset_filenames_for_ids(
                    manifests,
                    connected_dataset_ids,
                ),

            required_operations=
                operations,

            reasons=
                reasons,

            blockers=[],
        )
    )


# ============================================================
# REQUEST RESOLUTION
# ============================================================

def resolve_request(
    *,
    claim: VerifiedDocumentClaim,
    manifests: list[
        DatasetManifest
    ],
) -> RequestedAnalysisPlan:
    kind = (
        classify_request(
            claim
        )
    )


    transaction_dataset = (
        find_transaction_dataset(
            manifests
        )
    )


    transaction_ids: set[
        str
    ] = (
        {
            transaction_dataset.dataset_id
        }

        if transaction_dataset
        else
        set()
    )


    # ========================================================
    # UNKNOWN
    # ========================================================

    if (
        kind
        ==
        "unknown"
    ):
        return (
            RequestedAnalysisPlan(
                **base_plan_values(
                    claim
                ),

                kind=
                    kind,

                status=
                    "ambiguous",

                target_family=
                    None,

                matched_columns=[],

                required_dataset_ids=[],

                required_dataset_filenames=[],

                required_operations=[],

                reasons=[],

                blockers=[
                    (
                        "La demande est vérifiée "
                        "dans le document, mais le "
                        "planner déterministe ne "
                        "dispose pas encore d'une "
                        "règle suffisante pour la "
                        "traduire en plan analytique."
                    )
                ],
            )
        )


    # ========================================================
    # SIMPLE TRANSACTION COUNT
    # ========================================================

    if (
        kind
        ==
        "transaction_count"
    ):
        if transaction_dataset is None:
            return (
                finalize_plan(
                    claim=
                        claim,

                    kind=
                        kind,

                    target_family=
                        "descriptive_metric",

                    manifests=
                        manifests,

                    matches=[],

                    required_dataset_ids=
                        set(),

                    operations=[
                        (
                            "Compter les événements "
                            "transactionnels."
                        )
                    ],

                    reasons=[],

                    blockers=[
                        (
                            "Aucun dataset transactionnel "
                            "n'a été identifié avec une "
                            "confiance suffisante."
                        )
                    ],
                )
            )


        return (
            finalize_plan(
                claim=
                    claim,

                kind=
                    kind,

                target_family=
                    "descriptive_metric",

                manifests=
                    manifests,

                matches=[],

                required_dataset_ids=
                    transaction_ids,

                operations=[
                    (
                        "Compter les lignes ou "
                        "événements transactionnels "
                        "selon le grain vérifié du "
                        "dataset."
                    )
                ],

                reasons=[
                    (
                        f"{transaction_dataset.filename} "
                        "a été identifié comme dataset "
                        "transactionnel."
                    )
                ],

                blockers=[],
            )
        )


    # ========================================================
    # COMMON COLUMN MATCHES
    # ========================================================

    amount = (
        best_column_match(
            manifests,
            "amount",
        )
    )


    category = (
        best_column_match(
            manifests,
            "category",
        )
    )


    customer_id = (
        best_column_match(
            manifests,
            "customer_id",
            preferred_dataset_ids=
                transaction_ids,
        )
    )


    product_id = (
        best_column_match(
            manifests,
            "product_id",
            preferred_dataset_ids=
                transaction_ids,
        )
    )


    session_id = (
        best_column_match(
            manifests,
            "session_id",
            preferred_dataset_ids=
                transaction_ids,
        )
    )


    time_column = (
        best_column_match(
            manifests,
            "time",
            preferred_dataset_ids=
                transaction_ids,
        )
    )


    gender = (
        best_column_match(
            manifests,
            "gender",
        )
    )


    # ========================================================
    # REVENUE MOVING AVERAGE
    # ========================================================

    if (
        kind
        ==
        "revenue_moving_average"
    ):
        matches = [
            match

            for match
            in [
                amount,
                time_column,
            ]

            if match
            is not None
        ]


        blockers = []


        if transaction_dataset is None:
            blockers.append(
                (
                    "Aucun dataset transactionnel "
                    "fiable n'a été identifié."
                )
            )


        if amount is None:
            blockers.append(
                (
                    "Aucune variable monétaire "
                    "fiable n'a été identifiée."
                )
            )


        if time_column is None:
            blockers.append(
                (
                    "Aucune variable temporelle "
                    "d'achat n'a été identifiée."
                )
            )


        return (
            finalize_plan(
                claim=
                    claim,

                kind=
                    kind,

                target_family=
                    "time_series",

                manifests=
                    manifests,

                matches=
                    matches,

                required_dataset_ids=
                    transaction_ids,

                operations=[
                    (
                        "Relier la valeur monétaire "
                        "aux transactions si elle "
                        "provient d'un autre dataset."
                    ),
                    (
                        "Calculer le chiffre d'affaires "
                        "par période."
                    ),
                    (
                        "Calculer une moyenne mobile "
                        "sur une fenêtre explicitement "
                        "choisie ou configurée."
                    ),
                ],

                reasons=[
                    (
                        "La demande contient une "
                        "mesure monétaire, une "
                        "dimension temporelle et une "
                        "moyenne mobile."
                    )
                ],

                blockers=
                    blockers,
            )
        )


    # ========================================================
    # REVENUE BY CATEGORY
    # ========================================================

    if (
        kind
        ==
        "revenue_by_category"
    ):
        matches = [
            match

            for match
            in [
                amount,
                category,
            ]

            if match
            is not None
        ]


        blockers = []


        if transaction_dataset is None:
            blockers.append(
                (
                    "Aucun dataset transactionnel "
                    "fiable n'a été identifié."
                )
            )


        if amount is None:
            blockers.append(
                (
                    "Aucune variable monétaire "
                    "fiable n'a été identifiée."
                )
            )


        if category is None:
            blockers.append(
                (
                    "Aucune catégorie produit "
                    "fiable n'a été identifiée."
                )
            )


        return (
            finalize_plan(
                claim=
                    claim,

                kind=
                    kind,

                target_family=
                    "aggregate_breakdown",

                manifests=
                    manifests,

                matches=
                    matches,

                required_dataset_ids=
                    transaction_ids,

                operations=[
                    (
                        "Relier transactions et "
                        "référentiel produit si "
                        "nécessaire."
                    ),
                    (
                        "Calculer le chiffre "
                        "d'affaires transactionnel."
                    ),
                    (
                        "Agréger le chiffre "
                        "d'affaires par catégorie."
                    ),
                ],

                reasons=[
                    (
                        "La demande combine une "
                        "mesure monétaire et une "
                        "dimension catégorielle."
                    )
                ],

                blockers=
                    blockers,
            )
        )


    # ========================================================
    # CUSTOMERS BY PERIOD
    # ========================================================

    if (
        kind
        ==
        "customers_by_period"
    ):
        matches = [
            match

            for match
            in [
                customer_id,
                time_column,
            ]

            if match
            is not None
        ]


        blockers = []


        if transaction_dataset is None:
            blockers.append(
                (
                    "Aucun dataset transactionnel "
                    "fiable n'a été identifié."
                )
            )


        if customer_id is None:
            blockers.append(
                (
                    "Aucun identifiant client "
                    "fiable n'a été identifié."
                )
            )


        if time_column is None:
            blockers.append(
                (
                    "Aucune date d'achat fiable "
                    "n'a été identifiée."
                )
            )


        return (
            finalize_plan(
                claim=
                    claim,

                kind=
                    kind,

                target_family=
                    "time_series",

                manifests=
                    manifests,

                matches=
                    matches,

                required_dataset_ids=
                    transaction_ids,

                operations=[
                    (
                        "Ramener les dates au mois."
                    ),
                    (
                        "Compter les clients "
                        "distincts par mois."
                    ),
                ],

                reasons=[
                    (
                        "Le calcul nécessite un "
                        "identifiant client et une "
                        "date de transaction."
                    )
                ],

                blockers=
                    blockers,
            )
        )


    # ========================================================
    # PRODUCTS SOLD
    # ========================================================

    if (
        kind
        ==
        "products_sold_count"
    ):
        matches = [
            match

            for match
            in [
                product_id
            ]

            if match
            is not None
        ]


        blockers = []


        if transaction_dataset is None:
            blockers.append(
                (
                    "Aucun dataset transactionnel "
                    "fiable n'a été identifié."
                )
            )


        if product_id is None:
            blockers.append(
                (
                    "Aucun identifiant produit "
                    "fiable n'a été identifié."
                )
            )


        return (
            finalize_plan(
                claim=
                    claim,

                kind=
                    kind,

                target_family=
                    "descriptive_metric",

                manifests=
                    manifests,

                matches=
                    matches,

                required_dataset_ids=
                    transaction_ids,

                operations=[
                    (
                        "Compter les occurrences "
                        "produit dans les événements "
                        "transactionnels au grain "
                        "vérifié."
                    )
                ],

                reasons=[
                    (
                        "Le dataset transactionnel "
                        "permet de compter les "
                        "produits effectivement "
                        "observés dans les achats."
                    )
                ],

                blockers=
                    blockers,
            )
        )


    # ========================================================
    # TOP / FLOP PRODUCTS
    #
    # IMPORTANT:
    #
    # "top" and "flop" identify a ranking intention,
    # but they do not define the ranking metric.
    #
    # Revenue, units sold, transaction count or another
    # business metric could all be legitimate.
    #
    # DataLens must not choose one implicitly.
    # ========================================================

    if (
        kind
        in {
            "top_products",
            "flop_products",
        }
    ):
        matches = [
            match

            for match
            in [
                product_id,
                amount,
            ]

            if match
            is not None
        ]


        dataset_ids = list(
            dict.fromkeys(
                match.dataset_id

                for match
                in matches
            )
        )


        if (
            transaction_dataset
            is not None
            and
            transaction_dataset.dataset_id
            not in dataset_ids
        ):
            dataset_ids.append(
                transaction_dataset.dataset_id
            )


        direction = (
            "décroissant"

            if (
                kind
                ==
                "top_products"
            )

            else
            "croissant"
        )


        return (
            RequestedAnalysisPlan(
                **base_plan_values(
                    claim
                ),

                kind=
                    kind,

                status=
                    "ambiguous",

                target_family=
                    "ranking",

                matched_columns=[
                    public_match(
                        match
                    )

                    for match
                    in deduplicate_matches(
                        matches
                    )
                ],

                required_dataset_ids=
                    dataset_ids,

                required_dataset_filenames=
                    dataset_filenames_for_ids(
                        manifests,
                        dataset_ids,
                    ),

                required_operations=[
                    (
                        "Identifier la métrique "
                        "métier utilisée pour "
                        "classer les références."
                    ),
                    (
                        "Agréger cette métrique "
                        "par référence."
                    ),
                    (
                        f"Classer les références "
                        f"par ordre {direction}."
                    ),
                ],

                reasons=[
                    (
                        "La documentation demande "
                        "un classement des "
                        "références."
                    ),
                    (
                        "Les données permettent "
                        "d'identifier les références "
                        "et offrent au moins une "
                        "métrique candidate."
                    ),
                ],

                blockers=[
                    (
                        "La documentation ne précise "
                        "pas la métrique définissant "
                        "un top ou un flop. DataLens "
                        "refuse de choisir "
                        "implicitement entre chiffre "
                        "d'affaires, volume vendu, "
                        "nombre de transactions ou "
                        "une autre métrique métier."
                    )
                ],
            )
        )


    # ========================================================
    # PRODUCT CATEGORY DISTRIBUTION
    # ========================================================

    if (
        kind
        ==
        "product_category_distribution"
    ):
        matches = [
            match

            for match
            in [
                product_id,
                category,
            ]

            if match
            is not None
        ]


        blockers = []


        if product_id is None:
            blockers.append(
                (
                    "Aucun identifiant de référence "
                    "produit fiable n'a été identifié."
                )
            )


        if category is None:
            blockers.append(
                (
                    "Aucune catégorie produit "
                    "fiable n'a été identifiée."
                )
            )


        return (
            finalize_plan(
                claim=
                    claim,

                kind=
                    kind,

                target_family=
                    "categorical_breakdown",

                manifests=
                    manifests,

                matches=
                    matches,

                required_dataset_ids=
                    set(),

                operations=[
                    (
                        "Compter les références "
                        "distinctes par catégorie."
                    )
                ],

                reasons=[
                    (
                        "Le contexte documentaire "
                        "porte sur la répartition "
                        "des références."
                    )
                ],

                blockers=
                    blockers,
            )
        )


    # ========================================================
    # B2B REVENUE
    # ========================================================

    if (
        kind
        ==
        "b2b_revenue_distribution"
    ):
        explicit_b2b = (
            best_column_match(
                manifests,
                "b2b_explicit",
            )
        )


        generic_customer_type = (
            best_column_match(
                manifests,
                "customer_type",
            )
        )


        matches = [
            match

            for match
            in [
                explicit_b2b,
                generic_customer_type,
                customer_id,
                amount,
            ]

            if match
            is not None
        ]


        if (
            explicit_b2b
            is None
            and
            generic_customer_type
            is not None
        ):
            dataset_ids = list(
                dict.fromkeys(
                    match.dataset_id

                    for match
                    in matches
                )
            )


            return (
                RequestedAnalysisPlan(
                    **base_plan_values(
                        claim
                    ),

                    kind=
                        kind,

                    status=
                        "ambiguous",

                    target_family=
                        "aggregate_breakdown",

                    matched_columns=[
                        public_match(
                            match
                        )

                        for match
                        in deduplicate_matches(
                            matches
                        )
                    ],

                    required_dataset_ids=
                        dataset_ids,

                    required_dataset_filenames=
                        dataset_filenames_for_ids(
                            manifests,
                            dataset_ids,
                        ),

                    required_operations=[
                        (
                            "Vérifier les modalités "
                            "de la variable de type "
                            "client avant toute "
                            "classification BtoB."
                        )
                    ],

                    reasons=[],

                    blockers=[
                        (
                            "Une variable de segmentation "
                            "client existe peut-être, mais "
                            "le manifest ne permet pas de "
                            "confirmer qu'elle contient "
                            "explicitement une modalité BtoB."
                        )
                    ],
                )
            )


        blockers = []


        if explicit_b2b is None:
            blockers.append(
                (
                    "Aucune variable explicite "
                    "permettant d'identifier les "
                    "clients BtoB n'a été trouvée. "
                    "DataLens refuse d'inférer BtoB "
                    "à partir du montant d'achat ou "
                    "d'un comportement atypique."
                )
            )


        if amount is None:
            blockers.append(
                (
                    "Aucune variable monétaire "
                    "fiable n'a été identifiée."
                )
            )


        return (
            finalize_plan(
                claim=
                    claim,

                kind=
                    kind,

                target_family=
                    "aggregate_breakdown",

                manifests=
                    manifests,

                matches=
                    matches,

                required_dataset_ids=
                    transaction_ids,

                operations=[
                    (
                        "Identifier les clients BtoB "
                        "uniquement à partir d'une "
                        "variable explicite."
                    ),
                    (
                        "Calculer le chiffre d'affaires "
                        "par client."
                    ),
                    (
                        "Comparer la contribution BtoB "
                        "au chiffre d'affaires total."
                    ),
                ],

                reasons=[],

                blockers=
                    blockers,
            )
        )


    # ========================================================
    # LORENZ CURVE
    # ========================================================

    if (
        kind
        ==
        "lorenz_curve"
    ):
        matches = [
            match

            for match
            in [
                customer_id,
                amount,
            ]

            if match
            is not None
        ]


        blockers = []


        if transaction_dataset is None:
            blockers.append(
                (
                    "Aucun dataset transactionnel "
                    "fiable n'a été identifié."
                )
            )


        if customer_id is None:
            blockers.append(
                (
                    "Aucun identifiant client "
                    "fiable n'a été identifié."
                )
            )


        if amount is None:
            blockers.append(
                (
                    "Aucune mesure monétaire "
                    "fiable n'a été identifiée."
                )
            )


        return (
            finalize_plan(
                claim=
                    claim,

                kind=
                    kind,

                target_family=
                    "inequality",

                manifests=
                    manifests,

                matches=
                    matches,

                required_dataset_ids=
                    transaction_ids,

                operations=[
                    (
                        "Calculer le chiffre "
                        "d'affaires total par client."
                    ),
                    (
                        "Trier les clients par "
                        "contribution croissante."
                    ),
                    (
                        "Calculer les parts cumulées "
                        "de clients et de chiffre "
                        "d'affaires."
                    ),
                    (
                        "Construire la courbe "
                        "de Lorenz."
                    ),
                ],

                reasons=[
                    (
                        "Une courbe de Lorenz nécessite "
                        "une unité client et une mesure "
                        "positive agrégée par client."
                    )
                ],

                blockers=
                    blockers,
            )
        )


    # ========================================================
    # GENDER × CATEGORY
    # ========================================================

    if (
        kind
        ==
        "gender_category_association"
    ):
        matches = [
            match

            for match
            in [
                gender,
                category,
                customer_id,
                product_id,
            ]

            if match
            is not None
        ]


        blockers = []


        if gender is None:
            blockers.append(
                (
                    "Aucune variable de genre ou "
                    "sexe fiable n'a été identifiée."
                )
            )


        if category is None:
            blockers.append(
                (
                    "Aucune catégorie produit "
                    "fiable n'a été identifiée."
                )
            )


        if transaction_dataset is None:
            blockers.append(
                (
                    "Aucun dataset transactionnel "
                    "fiable n'a été identifié pour "
                    "relier clients et achats."
                )
            )


        return (
            finalize_plan(
                claim=
                    claim,

                kind=
                    kind,

                target_family=
                    "categorical_association",

                manifests=
                    manifests,

                matches=
                    matches,

                required_dataset_ids=
                    transaction_ids,

                operations=[
                    (
                        "Relier clients, transactions "
                        "et catégories produit."
                    ),
                    (
                        "Construire une table de "
                        "contingence genre × catégorie."
                    ),
                    (
                        "Laisser le moteur statistique "
                        "déterministe sélectionner "
                        "Khi² ou une alternative selon "
                        "les effectifs attendus."
                    ),
                ],

                reasons=[
                    (
                        "Les deux concepts demandés "
                        "sont catégoriels."
                    )
                ],

                blockers=
                    blockers,
            )
        )


    # ========================================================
    # AGE COMPONENTS
    # ========================================================

    (
        age_matches,
        age_operations,
        age_blockers,
    ) = resolve_age(
        manifests,
        transaction_dataset,
    )


    # ========================================================
    # AGE × TOTAL AMOUNT
    # ========================================================

    if (
        kind
        ==
        "age_total_amount_association"
    ):
        matches = [
            *age_matches,
            *[
                match

                for match
                in [
                    amount,
                    customer_id,
                ]

                if match
                is not None
            ],
        ]


        blockers = list(
            age_blockers
        )


        if amount is None:
            blockers.append(
                (
                    "Aucune variable monétaire "
                    "fiable n'a été identifiée."
                )
            )


        if customer_id is None:
            blockers.append(
                (
                    "Aucun identifiant client "
                    "fiable n'a été identifié."
                )
            )


        if transaction_dataset is None:
            blockers.append(
                (
                    "Aucun dataset transactionnel "
                    "fiable n'a été identifié."
                )
            )


        return (
            finalize_plan(
                claim=
                    claim,

                kind=
                    kind,

                target_family=
                    "quantitative_association",

                manifests=
                    manifests,

                matches=
                    matches,

                required_dataset_ids=
                    transaction_ids,

                operations=[
                    *age_operations,
                    (
                        "Calculer le montant total "
                        "des achats par client."
                    ),
                    (
                        "Analyser l'association entre "
                        "âge et montant total avec le "
                        "moteur statistique "
                        "déterministe."
                    ),
                ],

                reasons=[
                    (
                        "La demande relie deux "
                        "quantités au niveau client."
                    )
                ],

                blockers=
                    blockers,
            )
        )


    # ========================================================
    # AGE × FREQUENCY
    # ========================================================

    if (
        kind
        ==
        "age_frequency_association"
    ):
        matches = [
            *age_matches,
            *[
                match

                for match
                in [
                    customer_id,
                    session_id,
                ]

                if match
                is not None
            ],
        ]


        blockers = list(
            age_blockers
        )


        if customer_id is None:
            blockers.append(
                (
                    "Aucun identifiant client "
                    "fiable n'a été identifié."
                )
            )


        if session_id is None:
            blockers.append(
                (
                    "Aucun identifiant de session "
                    "ou commande fiable n'a été "
                    "identifié pour mesurer la "
                    "fréquence d'achat sans "
                    "confondre fréquence et nombre "
                    "de lignes transactionnelles."
                )
            )


        if transaction_dataset is None:
            blockers.append(
                (
                    "Aucun dataset transactionnel "
                    "fiable n'a été identifié."
                )
            )


        return (
            finalize_plan(
                claim=
                    claim,

                kind=
                    kind,

                target_family=
                    "quantitative_association",

                manifests=
                    manifests,

                matches=
                    matches,

                required_dataset_ids=
                    transaction_ids,

                operations=[
                    *age_operations,
                    (
                        "Compter les sessions ou "
                        "commandes distinctes par "
                        "client."
                    ),
                    (
                        "Analyser l'association entre "
                        "âge et fréquence d'achat."
                    ),
                ],

                reasons=[
                    (
                        "La fréquence est dérivée "
                        "au niveau client avant le "
                        "test d'association."
                    )
                ],

                blockers=
                    blockers,
            )
        )


    # ========================================================
    # AGE × AVERAGE BASKET
    # ========================================================

    if (
        kind
        ==
        "age_average_basket_association"
    ):
        matches = [
            *age_matches,
            *[
                match

                for match
                in [
                    customer_id,
                    session_id,
                    amount,
                ]

                if match
                is not None
            ],
        ]


        blockers = list(
            age_blockers
        )


        if customer_id is None:
            blockers.append(
                (
                    "Aucun identifiant client "
                    "fiable n'a été identifié."
                )
            )


        if session_id is None:
            blockers.append(
                (
                    "Aucun identifiant de session "
                    "ou commande fiable n'a été "
                    "identifié pour construire "
                    "un panier."
                )
            )


        if amount is None:
            blockers.append(
                (
                    "Aucune variable monétaire "
                    "fiable n'a été identifiée."
                )
            )


        if transaction_dataset is None:
            blockers.append(
                (
                    "Aucun dataset transactionnel "
                    "fiable n'a été identifié."
                )
            )


        return (
            finalize_plan(
                claim=
                    claim,

                kind=
                    kind,

                target_family=
                    "quantitative_association",

                manifests=
                    manifests,

                matches=
                    matches,

                required_dataset_ids=
                    transaction_ids,

                operations=[
                    *age_operations,
                    (
                        "Calculer le montant de "
                        "chaque panier ou session."
                    ),
                    (
                        "Calculer le panier moyen "
                        "par client."
                    ),
                    (
                        "Analyser l'association entre "
                        "âge et panier moyen."
                    ),
                ],

                reasons=[
                    (
                        "Le panier moyen doit être "
                        "dérivé au bon grain avant "
                        "l'analyse statistique."
                    )
                ],

                blockers=
                    blockers,
            )
        )


    # ========================================================
    # AGE × CATEGORY
    # ========================================================

    if (
        kind
        ==
        "age_category_association"
    ):
        matches = [
            *age_matches,
            *[
                match

                for match
                in [
                    category,
                    customer_id,
                    product_id,
                ]

                if match
                is not None
            ],
        ]


        blockers = list(
            age_blockers
        )


        if category is None:
            blockers.append(
                (
                    "Aucune catégorie produit "
                    "fiable n'a été identifiée."
                )
            )


        if transaction_dataset is None:
            blockers.append(
                (
                    "Aucun dataset transactionnel "
                    "fiable n'a été identifié pour "
                    "relier âge et catégories "
                    "achetées."
                )
            )


        return (
            finalize_plan(
                claim=
                    claim,

                kind=
                    kind,

                target_family=
                    "mixed_association",

                manifests=
                    manifests,

                matches=
                    matches,

                required_dataset_ids=
                    transaction_ids,

                operations=[
                    *age_operations,
                    (
                        "Relier chaque achat à la "
                        "catégorie produit."
                    ),
                    (
                        "Préserver l'âge quantitatif "
                        "dans les données préparées."
                    ),
                    (
                        "Laisser la politique "
                        "statistique déterministe "
                        "choisir entre comparaison "
                        "quantitative × catégorie "
                        "ou représentation par "
                        "tranches d'âge documentées."
                    ),
                ],

                reasons=[
                    (
                        "La demande combine une "
                        "variable d'âge et une "
                        "variable catégorielle."
                    )
                ],

                blockers=
                    blockers,
            )
        )


    # ========================================================
    # FALLBACK
    # ========================================================

    return (
        RequestedAnalysisPlan(
            **base_plan_values(
                claim
            ),

            kind=
                kind,

            status=
                "ambiguous",

            target_family=
                None,

            matched_columns=[],

            required_dataset_ids=[],

            required_dataset_filenames=[],

            required_operations=[],

            reasons=[],

            blockers=[
                (
                    "Aucune règle de résolution "
                    "déterministe n'est encore "
                    "disponible pour cette demande."
                )
            ],
        )
    )


# ============================================================
# REPORT
# ============================================================

def build_requested_analysis_plan(
    *,
    ingestion: MultiDatasetIngestion,

    analytical_requests: list[
        VerifiedDocumentClaim
    ],
) -> RequestedAnalysisPlanReport:
    manifests = list(
        ingestion.datasets
    )


    plans: list[
        RequestedAnalysisPlan
    ] = []


    for claim in analytical_requests:
        if (
            claim.category
            !=
            "analytical_request"
        ):
            continue


        plans.append(
            resolve_request(
                claim=
                    claim,

                manifests=
                    manifests,
            )
        )


    ready_count = sum(
        1

        for plan
        in plans

        if (
            plan.status
            ==
            "ready"
        )
    )


    blocked_count = sum(
        1

        for plan
        in plans

        if (
            plan.status
            ==
            "blocked"
        )
    )


    ambiguous_count = sum(
        1

        for plan
        in plans

        if (
            plan.status
            ==
            "ambiguous"
        )
    )


    return (
        RequestedAnalysisPlanReport(
            request_count=
                len(
                    plans
                ),

            ready_count=
                ready_count,

            blocked_count=
                blocked_count,

            ambiguous_count=
                ambiguous_count,

            requests=
                plans,

            planner_notes=[
                (
                    "Toutes les demandes analytiques "
                    "documentaires vérifiées sont "
                    "conservées. Aucun plafond de "
                    "recommandation exploratoire "
                    "n'est appliqué."
                ),
                (
                    "Le statut ready signifie que "
                    "les données et opérations "
                    "nécessaires ont été résolues, "
                    "pas que l'analyse a déjà été "
                    "exécutée."
                ),
                (
                    "Les jointures ne sont acceptées "
                    "dans cette version que lorsqu'un "
                    "chemin déterministe par "
                    "identifiants compatibles peut "
                    "être construit."
                ),
                (
                    "DataLens ne déduit jamais un "
                    "statut BtoB à partir d'un montant "
                    "d'achat élevé ou d'un comportement "
                    "atypique."
                ),
                (
                    "DataLens ne choisit pas "
                    "implicitement la métrique "
                    "définissant un top ou un flop "
                    "lorsque le document ne la "
                    "précise pas."
                ),
            ],

            planner_rule_version=
                REQUEST_PLANNER_RULE_VERSION,
        )
    )
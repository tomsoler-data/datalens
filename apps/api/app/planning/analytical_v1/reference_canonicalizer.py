from __future__ import annotations

from typing import (
    Literal,
)

from pydantic import (
    BaseModel,
    ConfigDict,
)

from app.planning.analytical_v1.contract import (
    AnalyticalPlannerCandidate,
)

from app.planning.analytical_v1.input import (
    AnalyticalPlannerInput,
    AnalyticalPlannerRequirementInput,
)


# ============================================================
# VERSION
# ============================================================

ANALYTICAL_REFERENCE_CANONICALIZER_VERSION = (
    "analytical_reference_canonicalizer_v1.0"
)


# ============================================================
# ISSUE CODES
# ============================================================

CanonicalizationIssueCode = Literal[
    "unknown_requirement",
    "unknown_reference",
    "ambiguous_reference",
]


# ============================================================
# REWRITE
# ============================================================

class AnalyticalReferenceRewrite(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
    )

    requirement_id: str

    step_id: str

    tool_name: str

    argument_path: str

    original_reference: str

    canonical_reference: str


# ============================================================
# ISSUE
# ============================================================

class AnalyticalReferenceCanonicalizationIssue(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
    )

    code: CanonicalizationIssueCode

    requirement_id: (
        str
        | None
    )

    step_id: (
        str
        | None
    )

    tool_name: (
        str
        | None
    )

    argument_path: (
        str
        | None
    )

    reference: (
        str
        | None
    )

    candidates: list[
        str
    ]

    message: str


# ============================================================
# RESULT
# ============================================================

class AnalyticalReferenceCanonicalizationResult(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
    )

    canonicalizer_version: str

    safe: bool

    canonicalized_candidate: (
        AnalyticalPlannerCandidate
    )

    rewrites: list[
        AnalyticalReferenceRewrite
    ]

    issues: list[
        AnalyticalReferenceCanonicalizationIssue
    ]


# ============================================================
# REFERENCE CATALOG
# ============================================================

class _ReferenceCatalog(
    BaseModel
):
    """
    Internal trusted catalog.

    exact_references
        All analytical qualified_names visible to the planner.

    aliases
        Unqualified name -> one or more qualified references.

    Example:

        sales.revenue
        support.revenue

    gives:

        revenue -> [
            sales.revenue,
            support.revenue
        ]

    Therefore "revenue" is ambiguous and must never be
    canonicalized automatically.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    exact_references: set[
        str
    ]

    aliases: dict[
        str,
        list[str]
    ]


def _build_reference_catalog(
    requirement: AnalyticalPlannerRequirementInput,
) -> _ReferenceCatalog:

    exact_references: set[
        str
    ] = set()


    aliases: dict[
        str,
        list[str]
    ] = {}


    for column in (
        requirement.analytical_columns
    ):

        qualified_name_raw = (
            column.get(
                "qualified_name"
            )
        )


        if qualified_name_raw is None:
            continue


        qualified_name = (
            str(
                qualified_name_raw
            )
            .strip()
        )


        if not qualified_name:
            continue


        exact_references.add(
            qualified_name
        )


        unqualified_name = (
            qualified_name
            .rsplit(
                ".",
                1,
            )[
                -1
            ]
        )


        aliases.setdefault(
            unqualified_name,
            [],
        )


        aliases[
            unqualified_name
        ].append(
            qualified_name
        )


    # ========================================================
    # DEDUP + DETERMINISTIC ORDER
    # ========================================================

    normalized_aliases = {
        alias:
            sorted(
                set(
                    references
                )
            )

        for (
            alias,
            references,
        )
        in aliases.items()
    }


    return (
        _ReferenceCatalog(
            exact_references=(
                exact_references
            ),

            aliases=(
                normalized_aliases
            ),
        )
    )


# ============================================================
# ISSUE HELPERS
# ============================================================

def _append_unknown_reference_issue(
    *,
    issues: list[
        AnalyticalReferenceCanonicalizationIssue
    ],
    requirement_id: str,
    step_id: str,
    tool_name: str,
    argument_path: str,
    reference: str,
) -> None:

    issues.append(
        AnalyticalReferenceCanonicalizationIssue(
            code=(
                "unknown_reference"
            ),

            requirement_id=(
                requirement_id
            ),

            step_id=(
                step_id
            ),

            tool_name=(
                tool_name
            ),

            argument_path=(
                argument_path
            ),

            reference=(
                reference
            ),

            candidates=[],

            message=(
                "Reference cannot be resolved "
                "deterministically from the analytical "
                "columns visible to this requirement: "
                f"{reference}"
            ),
        )
    )


def _append_ambiguous_reference_issue(
    *,
    issues: list[
        AnalyticalReferenceCanonicalizationIssue
    ],
    requirement_id: str,
    step_id: str,
    tool_name: str,
    argument_path: str,
    reference: str,
    candidates: list[str],
) -> None:

    issues.append(
        AnalyticalReferenceCanonicalizationIssue(
            code=(
                "ambiguous_reference"
            ),

            requirement_id=(
                requirement_id
            ),

            step_id=(
                step_id
            ),

            tool_name=(
                tool_name
            ),

            argument_path=(
                argument_path
            ),

            reference=(
                reference
            ),

            candidates=(
                candidates
            ),

            message=(
                "Reference is ambiguous and cannot be "
                "canonicalized automatically. "
                f"Reference={reference}, "
                f"candidates={candidates}"
            ),
        )
    )


# ============================================================
# REWRITE HELPER
# ============================================================

def _append_rewrite(
    *,
    rewrites: list[
        AnalyticalReferenceRewrite
    ],
    requirement_id: str,
    step_id: str,
    tool_name: str,
    argument_path: str,
    original_reference: str,
    canonical_reference: str,
) -> None:

    if (
        original_reference
        == canonical_reference
    ):
        return


    rewrites.append(
        AnalyticalReferenceRewrite(
            requirement_id=(
                requirement_id
            ),

            step_id=(
                step_id
            ),

            tool_name=(
                tool_name
            ),

            argument_path=(
                argument_path
            ),

            original_reference=(
                original_reference
            ),

            canonical_reference=(
                canonical_reference
            ),
        )
    )


# ============================================================
# SINGLE REFERENCE RESOLUTION
# ============================================================

def _canonicalize_reference(
    *,
    reference: str,
    catalog: _ReferenceCatalog,
    derived_outputs: set[str],
    requirement_id: str,
    step_id: str,
    tool_name: str,
    argument_path: str,
    rewrites: list[
        AnalyticalReferenceRewrite
    ],
    issues: list[
        AnalyticalReferenceCanonicalizationIssue
    ],
) -> str:
    """
    Deterministic reference resolution.

    Rules:

    1. A prior derived output is already valid.
    2. An exact qualified reference is already valid.
    3. A qualified but unknown reference is never repaired.
    4. An unqualified alias with exactly one match is repaired.
    5. Multiple matches -> ambiguous.
    6. Zero matches -> unknown.

    Examples:

        channel
            -> ad_performance.channel
            if unique

        region
            -> ambiguous
            if sales.region and customers.region exist

        sales.ghost
            -> unknown
            never suffix-matched to another column

        sum
            -> unknown
    """

    original_reference = (
        reference
    )


    normalized_reference = (
        reference.strip()
    )


    # ========================================================
    # DERIVED METRIC
    # ========================================================

    if (
        normalized_reference
        in derived_outputs
    ):

        _append_rewrite(
            rewrites=rewrites,
            requirement_id=requirement_id,
            step_id=step_id,
            tool_name=tool_name,
            argument_path=argument_path,
            original_reference=(
                original_reference
            ),
            canonical_reference=(
                normalized_reference
            ),
        )


        return (
            normalized_reference
        )


    # ========================================================
    # EXACT QUALIFIED REFERENCE
    # ========================================================

    if (
        normalized_reference
        in catalog.exact_references
    ):

        _append_rewrite(
            rewrites=rewrites,
            requirement_id=requirement_id,
            step_id=step_id,
            tool_name=tool_name,
            argument_path=argument_path,
            original_reference=(
                original_reference
            ),
            canonical_reference=(
                normalized_reference
            ),
        )


        return (
            normalized_reference
        )


    # ========================================================
    # UNKNOWN QUALIFIED REFERENCE
    #
    # Do not perform suffix guessing on an explicitly
    # qualified reference.
    # ========================================================

    if (
        "."
        in normalized_reference
    ):

        _append_unknown_reference_issue(
            issues=issues,
            requirement_id=requirement_id,
            step_id=step_id,
            tool_name=tool_name,
            argument_path=argument_path,
            reference=(
                normalized_reference
            ),
        )


        return (
            original_reference
        )


    # ========================================================
    # UNQUALIFIED ALIAS
    # ========================================================

    candidates = (
        catalog.aliases.get(
            normalized_reference,
            [],
        )
    )


    # ========================================================
    # UNIQUE MATCH
    # ========================================================

    if (
        len(
            candidates
        )
        == 1
    ):

        canonical_reference = (
            candidates[
                0
            ]
        )


        _append_rewrite(
            rewrites=rewrites,
            requirement_id=requirement_id,
            step_id=step_id,
            tool_name=tool_name,
            argument_path=argument_path,
            original_reference=(
                original_reference
            ),
            canonical_reference=(
                canonical_reference
            ),
        )


        return (
            canonical_reference
        )


    # ========================================================
    # AMBIGUOUS
    # ========================================================

    if (
        len(
            candidates
        )
        > 1
    ):

        _append_ambiguous_reference_issue(
            issues=issues,
            requirement_id=requirement_id,
            step_id=step_id,
            tool_name=tool_name,
            argument_path=argument_path,
            reference=(
                normalized_reference
            ),
            candidates=(
                candidates
            ),
        )


        return (
            original_reference
        )


    # ========================================================
    # UNKNOWN
    # ========================================================

    _append_unknown_reference_issue(
        issues=issues,
        requirement_id=requirement_id,
        step_id=step_id,
        tool_name=tool_name,
        argument_path=argument_path,
        reference=(
            normalized_reference
        ),
    )


    return (
        original_reference
    )


# ============================================================
# LIST OF REFERENCES
# ============================================================

def _canonicalize_reference_list(
    *,
    references: list[str],
    catalog: _ReferenceCatalog,
    derived_outputs: set[str],
    requirement_id: str,
    step_id: str,
    tool_name: str,
    argument_name: str,
    rewrites: list[
        AnalyticalReferenceRewrite
    ],
    issues: list[
        AnalyticalReferenceCanonicalizationIssue
    ],
) -> list[str]:

    result: list[
        str
    ] = []


    for (
        index,
        reference,
    ) in enumerate(
        references
    ):

        result.append(
            _canonicalize_reference(
                reference=(
                    reference
                ),

                catalog=(
                    catalog
                ),

                derived_outputs=(
                    derived_outputs
                ),

                requirement_id=(
                    requirement_id
                ),

                step_id=(
                    step_id
                ),

                tool_name=(
                    tool_name
                ),

                argument_path=(
                    f"{argument_name}[{index}]"
                ),

                rewrites=(
                    rewrites
                ),

                issues=(
                    issues
                ),
            )
        )


    return result


# ============================================================
# CANONICALIZE ONE ACTION
# ============================================================

def _canonicalize_action_payload(
    *,
    action_payload: dict,
    catalog: _ReferenceCatalog,
    derived_outputs: set[str],
    requirement_id: str,
    step_id: str,
    rewrites: list[
        AnalyticalReferenceRewrite
    ],
    issues: list[
        AnalyticalReferenceCanonicalizationIssue
    ],
) -> dict:

    action = dict(
        action_payload
    )


    tool_name = str(
        action[
            "name"
        ]
    )


    # ========================================================
    # AGGREGATE
    # ========================================================

    if (
        tool_name
        == "aggregate"
    ):

        action[
            "metrics"
        ] = (
            _canonicalize_reference_list(
                references=(
                    action[
                        "metrics"
                    ]
                ),

                catalog=(
                    catalog
                ),

                derived_outputs=(
                    derived_outputs
                ),

                requirement_id=(
                    requirement_id
                ),

                step_id=(
                    step_id
                ),

                tool_name=(
                    tool_name
                ),

                argument_name=(
                    "metrics"
                ),

                rewrites=(
                    rewrites
                ),

                issues=(
                    issues
                ),
            )
        )


        group_by = (
            action.get(
                "group_by"
            )
        )


        if group_by is not None:

            action[
                "group_by"
            ] = (
                _canonicalize_reference_list(
                    references=(
                        group_by
                    ),

                    catalog=(
                        catalog
                    ),

                    derived_outputs=(
                        derived_outputs
                    ),

                    requirement_id=(
                        requirement_id
                    ),

                    step_id=(
                        step_id
                    ),

                    tool_name=(
                        tool_name
                    ),

                    argument_name=(
                        "group_by"
                    ),

                    rewrites=(
                        rewrites
                    ),

                    issues=(
                        issues
                    ),
                )
            )


        return action


    # ========================================================
    # BUILD ENTITY VIEW
    # ========================================================

    if (
        tool_name
        == "build_entity_view"
    ):

        action[
            "entity"
        ] = (
            _canonicalize_reference(
                reference=(
                    action[
                        "entity"
                    ]
                ),

                catalog=(
                    catalog
                ),

                derived_outputs=(
                    derived_outputs
                ),

                requirement_id=(
                    requirement_id
                ),

                step_id=(
                    step_id
                ),

                tool_name=(
                    tool_name
                ),

                argument_path=(
                    "entity"
                ),

                rewrites=(
                    rewrites
                ),

                issues=(
                    issues
                ),
            )
        )


        return action


    # ========================================================
    # DERIVE METRIC
    #
    # Inputs are analytical references.
    #
    # output is NOT a source reference.
    #
    # formula is deliberately NOT rewritten here.
    # Formula safety belongs to a dedicated expression layer.
    # ========================================================

    if (
        tool_name
        == "derive_metric"
    ):

        action[
            "inputs"
        ] = (
            _canonicalize_reference_list(
                references=(
                    action[
                        "inputs"
                    ]
                ),

                catalog=(
                    catalog
                ),

                derived_outputs=(
                    derived_outputs
                ),

                requirement_id=(
                    requirement_id
                ),

                step_id=(
                    step_id
                ),

                tool_name=(
                    tool_name
                ),

                argument_name=(
                    "inputs"
                ),

                rewrites=(
                    rewrites
                ),

                issues=(
                    issues
                ),
            )
        )


        return action


    # ========================================================
    # DISTRIBUTION
    # ========================================================

    if (
        tool_name
        in {
            "analyze_distribution",
            "detect_outliers",
        }
    ):

        action[
            "target"
        ] = (
            _canonicalize_reference(
                reference=(
                    action[
                        "target"
                    ]
                ),

                catalog=(
                    catalog
                ),

                derived_outputs=(
                    derived_outputs
                ),

                requirement_id=(
                    requirement_id
                ),

                step_id=(
                    step_id
                ),

                tool_name=(
                    tool_name
                ),

                argument_path=(
                    "target"
                ),

                rewrites=(
                    rewrites
                ),

                issues=(
                    issues
                ),
            )
        )


        return action


    # ========================================================
    # ENTITY OUTLIERS
    # ========================================================

    if (
        tool_name
        == "detect_entity_outliers"
    ):

        action[
            "entity"
        ] = (
            _canonicalize_reference(
                reference=(
                    action[
                        "entity"
                    ]
                ),

                catalog=(
                    catalog
                ),

                derived_outputs=(
                    derived_outputs
                ),

                requirement_id=(
                    requirement_id
                ),

                step_id=(
                    step_id
                ),

                tool_name=(
                    tool_name
                ),

                argument_path=(
                    "entity"
                ),

                rewrites=(
                    rewrites
                ),

                issues=(
                    issues
                ),
            )
        )


        action[
            "metrics"
        ] = (
            _canonicalize_reference_list(
                references=(
                    action[
                        "metrics"
                    ]
                ),

                catalog=(
                    catalog
                ),

                derived_outputs=(
                    derived_outputs
                ),

                requirement_id=(
                    requirement_id
                ),

                step_id=(
                    step_id
                ),

                tool_name=(
                    tool_name
                ),

                argument_name=(
                    "metrics"
                ),

                rewrites=(
                    rewrites
                ),

                issues=(
                    issues
                ),
            )
        )


        return action


    # ========================================================
    # GROUP COMPARISON
    # ========================================================

    if (
        tool_name
        == "compare_groups"
    ):

        action[
            "target"
        ] = (
            _canonicalize_reference(
                reference=(
                    action[
                        "target"
                    ]
                ),

                catalog=(
                    catalog
                ),

                derived_outputs=(
                    derived_outputs
                ),

                requirement_id=(
                    requirement_id
                ),

                step_id=(
                    step_id
                ),

                tool_name=(
                    tool_name
                ),

                argument_path=(
                    "target"
                ),

                rewrites=(
                    rewrites
                ),

                issues=(
                    issues
                ),
            )
        )


        action[
            "group_by"
        ] = (
            _canonicalize_reference(
                reference=(
                    action[
                        "group_by"
                    ]
                ),

                catalog=(
                    catalog
                ),

                derived_outputs=(
                    derived_outputs
                ),

                requirement_id=(
                    requirement_id
                ),

                step_id=(
                    step_id
                ),

                tool_name=(
                    tool_name
                ),

                argument_path=(
                    "group_by"
                ),

                rewrites=(
                    rewrites
                ),

                issues=(
                    issues
                ),
            )
        )


        return action


    # ========================================================
    # ASSOCIATION
    # ========================================================

    if (
        tool_name
        == "measure_association"
    ):

        for argument_name in (
            "target",
            "value",
        ):

            action[
                argument_name
            ] = (
                _canonicalize_reference(
                    reference=(
                        action[
                            argument_name
                        ]
                    ),

                    catalog=(
                        catalog
                    ),

                    derived_outputs=(
                        derived_outputs
                    ),

                    requirement_id=(
                        requirement_id
                    ),

                    step_id=(
                        step_id
                    ),

                    tool_name=(
                        tool_name
                    ),

                    argument_path=(
                        argument_name
                    ),

                    rewrites=(
                        rewrites
                    ),

                    issues=(
                        issues
                    ),
                )
            )


        return action


    # ========================================================
    # TIME SERIES
    # ========================================================

    if (
        tool_name
        == "analyze_time_series"
    ):

        for argument_name in (
            "date",
            "target",
        ):

            action[
                argument_name
            ] = (
                _canonicalize_reference(
                    reference=(
                        action[
                            argument_name
                        ]
                    ),

                    catalog=(
                        catalog
                    ),

                    derived_outputs=(
                        derived_outputs
                    ),

                    requirement_id=(
                        requirement_id
                    ),

                    step_id=(
                        step_id
                    ),

                    tool_name=(
                        tool_name
                    ),

                    argument_path=(
                        argument_name
                    ),

                    rewrites=(
                        rewrites
                    ),

                    issues=(
                        issues
                    ),
                )
            )


        return action


    # ========================================================
    # CONTRACT CURRENTLY MAKES UNKNOWN TOOLS IMPOSSIBLE.
    #
    # Return unchanged defensively.
    # ========================================================

    return action


# ============================================================
# PUBLIC CANONICALIZER
# ============================================================

def canonicalize_analytical_references(
    *,
    candidate: AnalyticalPlannerCandidate,
    planner_input: AnalyticalPlannerInput,
) -> AnalyticalReferenceCanonicalizationResult:
    """
    Safely canonicalize analytical references.

    This function does NOT:

    - repair reasoning;
    - change intent;
    - change family;
    - change target_grain;
    - add/remove/reorder steps;
    - choose tools;
    - invent columns;
    - infer joins;
    - rewrite formulas.

    It only resolves unique aliases.
    """

    requirements_by_id = {
        requirement.requirement_id:
            requirement

        for requirement
        in planner_input.requirements
    }


    candidate_payload = (
        candidate.model_dump(
            mode="json",
        )
    )


    rewrites: list[
        AnalyticalReferenceRewrite
    ] = []


    issues: list[
        AnalyticalReferenceCanonicalizationIssue
    ] = []


    # ========================================================
    # PLAN BY PLAN
    # ========================================================

    for plan_payload in (
        candidate_payload[
            "plans"
        ]
    ):

        requirement_id = str(
            plan_payload[
                "requirement_id"
            ]
        )


        requirement = (
            requirements_by_id.get(
                requirement_id
            )
        )


        # ====================================================
        # UNKNOWN REQUIREMENT
        #
        # Do not attempt reference resolution without trusted
        # requirement context.
        # ====================================================

        if requirement is None:

            issues.append(
                AnalyticalReferenceCanonicalizationIssue(
                    code=(
                        "unknown_requirement"
                    ),

                    requirement_id=(
                        requirement_id
                    ),

                    step_id=None,

                    tool_name=None,

                    argument_path=None,

                    reference=None,

                    candidates=[],

                    message=(
                        "Cannot canonicalize references for "
                        "an unknown planner requirement: "
                        f"{requirement_id}"
                    ),
                )
            )


            continue


        catalog = (
            _build_reference_catalog(
                requirement
            )
        )


        # ====================================================
        # Derived outputs are available only after their
        # definition in the ordered plan.
        # ====================================================

        derived_outputs: set[
            str
        ] = set()


        for step_payload in (
            plan_payload[
                "steps"
            ]
        ):

            step_id = str(
                step_payload[
                    "step_id"
                ]
            )


            original_action = (
                step_payload[
                    "action"
                ]
            )


            canonical_action = (
                _canonicalize_action_payload(
                    action_payload=(
                        original_action
                    ),

                    catalog=(
                        catalog
                    ),

                    derived_outputs=(
                        derived_outputs
                    ),

                    requirement_id=(
                        requirement_id
                    ),

                    step_id=(
                        step_id
                    ),

                    rewrites=(
                        rewrites
                    ),

                    issues=(
                        issues
                    ),
                )
            )


            step_payload[
                "action"
            ] = (
                canonical_action
            )


            # =================================================
            # Make the derived metric available only AFTER the
            # step that defines it.
            # =================================================

            if (
                canonical_action[
                    "name"
                ]
                == "derive_metric"
            ):

                derived_outputs.add(
                    str(
                        canonical_action[
                            "output"
                        ]
                    )
                )


    # ========================================================
    # REVALIDATE AGAINST THE CONTRACT
    # ========================================================

    canonicalized_candidate = (
        AnalyticalPlannerCandidate
        .model_validate(
            candidate_payload
        )
    )


    return (
        AnalyticalReferenceCanonicalizationResult(
            canonicalizer_version=(
                ANALYTICAL_REFERENCE_CANONICALIZER_VERSION
            ),

            safe=(
                len(
                    issues
                )
                == 0
            ),

            canonicalized_candidate=(
                canonicalized_candidate
            ),

            rewrites=(
                rewrites
            ),

            issues=(
                issues
            ),
        )
    )


# ============================================================
# EXECUTION GUARD
# ============================================================

def require_safe_reference_canonicalization(
    *,
    candidate: AnalyticalPlannerCandidate,
    planner_input: AnalyticalPlannerInput,
) -> AnalyticalPlannerCandidate:
    """
    Return the canonicalized candidate only when every
    analytical reference was resolved deterministically.

    Unknown or ambiguous references are never guessed.
    """

    result = (
        canonicalize_analytical_references(
            candidate=candidate,
            planner_input=planner_input,
        )
    )


    if not result.safe:

        diagnostics = [
            (
                f"{issue.code}:"
                f"{issue.reference}"
            )

            for issue
            in result.issues
        ]


        raise ValueError(
            "Analytical reference canonicalization "
            "could not be completed safely. "
            f"Issues: {diagnostics}"
        )


    return (
        result.canonicalized_candidate
    )
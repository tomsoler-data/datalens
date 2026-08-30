from __future__ import annotations

import ast
import hashlib
import subprocess

from pathlib import Path


print(
    "=== DATALENS QLORA v0.4 HOTEL RUNNER STATIC TEST v0.1 ==="
)
print()


ROOT = Path(
    __file__
).resolve().parent


RUNNER = (
    ROOT
    /
    "app"
    /
    "adaptation"
    /
    "reasoning_evaluation_runner_v0_4.py"
)

HISTORICAL = (
    ROOT
    /
    "app"
    /
    "adaptation"
    /
    "reasoning_evaluation_runner_v0_2.py"
)


EXPECTED_HEAD = (
    "b82e89eadc423aadba247be226fdd7e3a96cc7e4"
)

EXPECTED_HISTORICAL_SHA = (
    "27c0fb102748da982c7635f8be29eac7"
    "93e96de148a30d8bb3c1f47c6908b860"
)


def expect(
    condition,
    message,
):
    if not condition:
        raise AssertionError(
            message
        )


def git(
    *args: str,
) -> str:
    result = subprocess.run(
        [
            "git",
            *args,
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr
        )

    return result.stdout.strip()


def sha256_file(
    path: Path,
) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def function_source(
    source: str,
    name: str,
) -> str:
    tree = ast.parse(
        source
    )

    matches = [
        node

        for node
        in tree.body

        if (
            isinstance(
                node,
                ast.FunctionDef,
            )
            and
            node.name
            ==
            name
        )
    ]

    expect(
        len(
            matches
        )
        ==
        1,
        (
            "Expected exactly one function: "
            f"{name}"
        ),
    )

    segment = ast.get_source_segment(
        source,
        matches[
            0
        ],
    )

    expect(
        segment is not None,
        (
            "Could not recover function source: "
            f"{name}"
        ),
    )

    return segment


# ============================================================
# GIT AUTHORITY
# ============================================================


expect(
    git(
        "rev-parse",
        "HEAD",
    )
    ==
    EXPECTED_HEAD,
    "Unexpected HEAD.",
)


expect(
    RUNNER.is_file(),
    "v0.4 Hotel runner missing.",
)


expect(
    HISTORICAL.is_file(),
    "Historical Hotel runner missing.",
)


expect(
    sha256_file(
        HISTORICAL
    )
    ==
    EXPECTED_HISTORICAL_SHA,
    "Historical Hotel runner SHA changed.",
)


source = RUNNER.read_text(
    encoding="utf-8-sig"
)

historical = HISTORICAL.read_text(
    encoding="utf-8-sig"
)

tree = ast.parse(
    source
)


# ============================================================
# VERSION / PATH AUTHORITY
# ============================================================


for value in (
    "adapted_reasoning_evaluation_runner_v0.4",
    "adapted_reasoning_evaluation_manifest_v0.4",
    "adapted_reasoning_evaluation_receipt_v0.4",
):
    expect(
        value in source,
        (
            "Missing v0.4 rule authority: "
            f"{value}"
        ),
    )


def assignment_string_literals(
    name: str,
) -> tuple[
    str,
    ...,
]:
    matches = []

    for node in tree.body:
        value_node = None

        if isinstance(
            node,
            ast.Assign,
        ):
            if (
                len(
                    node.targets
                )
                ==
                1
                and
                isinstance(
                    node.targets[
                        0
                    ],
                    ast.Name,
                )
                and
                node.targets[
                    0
                ].id
                ==
                name
            ):
                value_node = (
                    node.value
                )

        elif isinstance(
            node,
            ast.AnnAssign,
        ):
            if (
                isinstance(
                    node.target,
                    ast.Name,
                )
                and
                node.target.id
                ==
                name
            ):
                value_node = (
                    node.value
                )

        if value_node is not None:
            matches.append(
                value_node
            )

    expect(
        len(
            matches
        )
        ==
        1,
        (
            "Expected exactly one assignment: "
            f"{name}"
        ),
    )

    return tuple(
        item.value

        for item
        in ast.walk(
            matches[
                0
            ]
        )

        if (
            isinstance(
                item,
                ast.Constant,
            )
            and
            isinstance(
                item.value,
                str,
            )
        )
    )


expected_path_fragments = {
    "MANIFEST_PATH": (
        (
            "datalens_semantic_qlora_v0.4_"
            "reasoning_evaluation_v0.1_manifest.json"
        ),
    ),

    "MANIFEST_FREEZE_PATH": (
        (
            "datalens_semantic_qlora_v0.4_"
            "reasoning_evaluation_v0.1_manifest_freeze.json"
        ),
    ),

    "REPORT_PATH": (
        (
            "datalens_semantic_qlora_v0.4_"
            "reasoning_evaluation_v0.1_report.json"
        ),
    ),

    "RECEIPT_PATH": (
        (
            "datalens_semantic_qlora_v0.4_"
            "reasoning_evaluation_v0.1_receipt.json"
        ),
    ),

    "ADAPTER_PATH": (
        "artifacts",
        "adaptation",
        "adapters",
        "datalens_semantic_qlora_v0.4_adapter",
    ),

    "RUNNER_REPO_PATH": (
        (
            "apps/api/app/adaptation/"
            "reasoning_evaluation_runner_v0_4.py"
        ),
    ),

    "TEST_REPO_PATH": (
        (
            "apps/api/"
            "test_reasoning_evaluation_runner_v0_4_v0_1.py"
        ),
    ),

    "MANIFEST_REPO_PATH": (
        (
            "apps/api/artifacts/adaptation/evaluation/"
            "datalens_semantic_qlora_v0.4_"
            "reasoning_evaluation_v0.1_manifest.json"
        ),
    ),

    "MANIFEST_FREEZE_REPO_PATH": (
        (
            "apps/api/artifacts/adaptation/evaluation/"
            "datalens_semantic_qlora_v0.4_"
            "reasoning_evaluation_v0.1_manifest_freeze.json"
        ),
    ),
}


for (
    assignment_name,
    required_fragments,
) in expected_path_fragments.items():
    literals = (
        assignment_string_literals(
            assignment_name
        )
    )

    for fragment in required_fragments:
        expect(
            fragment in literals,
            (
                "Missing v0.4 path authority fragment.\n"
                f"Assignment: {assignment_name}\n"
                f"Fragment:   {fragment}\n"
                f"Literals:   {literals}"
            ),
        )


expect(
    (
        "datalens-semantic-qlora-v0.3-training-v0.1"
        not in source
    ),
    "Historical v0.3 adapter cache path survived port.",
)


print(
    "v0.4 rule versions exact: PASS"
)

print(
    "v0.4 manifest path AST binding: PASS"
)

print(
    "v0.4 manifest freeze path AST binding: PASS"
)

print(
    "v0.4 report path AST binding: PASS"
)

print(
    "v0.4 receipt path AST binding: PASS"
)

print(
    "v0.4 adapter path AST binding: PASS"
)

print(
    "v0.4 runner/test repository paths: PASS"
)


# ============================================================
# V0.4 PREREQUISITES
# ============================================================


def assignment_literal(
    name: str,
):
    matches = []

    for node in tree.body:
        value_node = None

        if isinstance(
            node,
            ast.Assign,
        ):
            if (
                len(
                    node.targets
                )
                ==
                1
                and
                isinstance(
                    node.targets[
                        0
                    ],
                    ast.Name,
                )
                and
                node.targets[
                    0
                ].id
                ==
                name
            ):
                value_node = (
                    node.value
                )

        elif isinstance(
            node,
            ast.AnnAssign,
        ):
            if (
                isinstance(
                    node.target,
                    ast.Name,
                )
                and
                node.target.id
                ==
                name
            ):
                value_node = (
                    node.value
                )

        if value_node is not None:
            matches.append(
                value_node
            )

    expect(
        len(
            matches
        )
        ==
        1,
        (
            "Expected exactly one assignment: "
            f"{name}"
        ),
    )

    try:
        return ast.literal_eval(
            matches[
                0
            ]
        )

    except Exception as exc:
        raise AssertionError(
            (
                "Assignment is not a literal value: "
                f"{name}"
            )
        ) from exc


expected_prerequisite_constants = {
    "EXPECTED_HISTORICAL_SCORING_RUNNER_SHA256":
        (
            "27c0fb102748da982c7635f8be29eac7"
            "93e96de148a30d8bb3c1f47c6908b860"
        ),

    "EXPECTED_TRAINING_REPORT_SHA256":
        (
            "759ba4957806daab8b7a14d3aeb2b068"
            "59e0bcd6193d30cb877b63748617e04d"
        ),

    "EXPECTED_TRAINING_RECEIPT_SHA256":
        (
            "f412062f78432d7c432d4b36beed9d84"
            "d527d5990279240030a0a31227dffaee"
        ),

    "EXPECTED_ADAPTER_BUNDLE_SHA256":
        (
            "0351980df6d86096195c0971deb30c725"
            "e155c71aa5de8054b2b37fa42090716"
        ),

    "EXPECTED_ADAPTER_CONFIG_SHA256":
        (
            "3ae14896612f6bf74ee7786a450e2ac0"
            "f08f3da9f33391505cb1a7dc823dcdb8"
        ),

    "EXPECTED_ADAPTER_WEIGHTS_SHA256":
        (
            "4f145b0bf37f67841c09f02b86679634"
            "a9532491d2f560b0e7c5c328009e4610"
        ),

    "EXPECTED_ADAPTER_README_SHA256":
        (
            "6ecdbb662eaed8010ab0e012a2b95b79"
            "543884cf294406dc6da2cde64f98389d"
        ),

    "EXPECTED_S3_REPORT_SHA256":
        (
            "b0b662c31f7bbc013968cd7a69968aba"
            "21621c13cfe8de72bef2f4ddde0c1e6c"
        ),

    "EXPECTED_S3_RECEIPT_SHA256":
        (
            "fb4d81feb1c20c77e8fe3eaaf491ca7f"
            "831c32c96a6dcf29389e7b06dae28b96"
        ),
}


for (
    constant_name,
    expected_value,
) in expected_prerequisite_constants.items():
    actual_value = (
        assignment_literal(
            constant_name
        )
    )

    expect(
        actual_value
        ==
        expected_value,
        (
            "Prerequisite authority mismatch.\n"
            f"Constant: {constant_name}\n"
            f"Expected: {expected_value}\n"
            f"Actual:   {actual_value}"
        ),
    )


function_names = {
    node.name

    for node
    in tree.body

    if isinstance(
        node,
        (
            ast.FunctionDef,
            ast.AsyncFunctionDef,
        ),
    )
}


expect(
    (
        "validate_v0_4_diagnostic_prerequisites"
        in
        function_names
    ),
    "validate_v0_4_diagnostic_prerequisites() missing.",
)


validate_node = next(
    (
        node

        for node
        in tree.body

        if (
            isinstance(
                node,
                ast.FunctionDef,
            )
            and
            node.name
            ==
            "validate_static_contract"
        )
    ),
    None,
)


expect(
    validate_node is not None,
    "validate_static_contract() missing.",
)


body = list(
    validate_node.body
)


expect(
    bool(
        body
    ),
    "validate_static_contract() has no body.",
)


first_index = 0


if (
    isinstance(
        body[
            0
        ],
        ast.Expr,
    )
    and
    isinstance(
        body[
            0
        ].value,
        ast.Constant,
    )
    and
    isinstance(
        body[
            0
        ].value.value,
        str,
    )
):
    first_index = 1


expect(
    first_index
    <
    len(
        body
    ),
    "No executable static-contract action found.",
)


first = body[
    first_index
]


expect(
    (
        isinstance(
            first,
            ast.Expr,
        )
        and
        isinstance(
            first.value,
            ast.Call,
        )
        and
        isinstance(
            first.value.func,
            ast.Name,
        )
        and
        first.value.func.id
        ==
        "validate_v0_4_diagnostic_prerequisites"
    ),
    (
        "v0.4 prerequisite gate is not the first "
        "static-contract action."
    ),
)


print(
    "Historical scoring runner authority exact: PASS"
)

print(
    "Official training report authority exact: PASS"
)

print(
    "Official training receipt authority exact: PASS"
)

print(
    "Official adapter bundle authority exact: PASS"
)

print(
    "Official adapter file authorities exact: PASS"
)

print(
    "Official S3 report authority exact: PASS"
)

print(
    "Official S3 receipt authority exact: PASS"
)

print(
    "v0.4 prerequisite validator present: PASS"
)

print(
    "v0.4 prerequisite validator runs first: PASS"
)


# ============================================================
# FROZEN SCORING ENGINE
# ============================================================


for function_name in (
    "_to_token_list",
    "_encode_candidate",
    "_score_case",
    "_evaluate_candidate_model",
    "_paired_comparison",
):
    expect(
        (
            function_source(
                source,
                function_name,
            )
            ==
            function_source(
                historical,
                function_name,
            )
        ),
        (
            "Historical scoring engine changed: "
            f"{function_name}"
        ),
    )


# ============================================================
# NO FREE GENERATION
# ============================================================


generation_lines = []


for node in ast.walk(
    tree
):
    if not isinstance(
        node,
        ast.Call,
    ):
        continue

    if (
        isinstance(
            node.func,
            ast.Attribute,
        )
        and
        node.func.attr
        ==
        "generate"
    ):
        generation_lines.append(
            node.lineno
        )


expect(
    not generation_lines,
    (
        "Hotel diagnostic contains free generation: "
        f"{generation_lines}"
    ),
)


# ============================================================
# HEAVY ML IMPORTS MUST REMAIN DEFERRED
# ============================================================


top_level_modules = []


for node in tree.body:
    if isinstance(
        node,
        ast.Import,
    ):
        top_level_modules.extend(
            alias.name
            for alias in node.names
        )

    elif isinstance(
        node,
        ast.ImportFrom,
    ):
        top_level_modules.append(
            node.module
            or
            ""
        )


for forbidden in (
    "torch",
    "transformers",
    "peft",
    "bitsandbytes",
):
    expect(
        not any(
            module == forbidden
            or
            module.startswith(
                forbidden
                +
                "."
            )
            for module in top_level_modules
        ),
        (
            "Heavy ML dependency imported "
            "before authorization: "
            f"{forbidden}"
        ),
    )


# ============================================================
# DEFERRED EVALUATION BOUNDARY
# ============================================================


lower_source = source.lower()


for forbidden in (
    "airport",
    "greenhouse",
):
    expect(
        forbidden
        not in
        lower_source,
        (
            "Deferred evaluation leaked into "
            "Hotel runner: "
            f"{forbidden}"
        ),
    )


# ============================================================
# RESULT FILES MUST NOT EXIST
# ============================================================


report = (
    ROOT
    /
    "artifacts"
    /
    "adaptation"
    /
    "evaluation"
    /
    "datalens_semantic_qlora_v0.4_reasoning_evaluation_v0.1_report.json"
)

receipt = (
    ROOT
    /
    "artifacts"
    /
    "adaptation"
    /
    "evaluation"
    /
    "datalens_semantic_qlora_v0.4_reasoning_evaluation_v0.1_receipt.json"
)


expect(
    not report.exists(),
    "v0.4 Hotel report already exists.",
)

expect(
    not receipt.exists(),
    "v0.4 Hotel receipt already exists.",
)


print(
    "Historical scoring runner SHA exact: PASS"
)

print(
    "v0.4 adapter authority: PASS"
)

print(
    "Official S3 prerequisite authority: PASS"
)

print(
    "Official training authority: PASS"
)

print(
    "Five scoring functions unchanged: PASS"
)

print(
    "Teacher-forced scoring preserved: PASS"
)

print(
    "Free generation used: False"
)

print(
    "Heavy ML imports deferred: PASS"
)

print(
    "Airport dependency: False"
)

print(
    "Greenhouse dependency: False"
)

print(
    "Hotel inference executed: False"
)

print(
    "DATALENS QLORA v0.4 HOTEL RUNNER STATIC TEST v0.1: PASS"
)

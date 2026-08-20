from __future__ import annotations

import ast

from pathlib import Path


# ============================================================
# VERSION
# ============================================================

PROMOTION_VERSION = (
    "analytical_planner_prompt_production_promotion_v1.0"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = (
    Path(
        __file__
    )
    .resolve()
    .parents[2]
)


SOURCE_PATH = (
    BASE_DIR
    / "app"
    / "evals"
    / "analytical_planner_model_runner_v0_9.py"
)


TARGET_PATH = (
    BASE_DIR
    / "app"
    / "ai"
    / "analytical_planner_prompt_v1.py"
)


# ============================================================
# SOURCE CONSTANTS
# ============================================================

SOURCE_PROMPT_NAME = (
    "SYSTEM_PROMPT_V0_9"
)


SOURCE_VERSION_NAME = (
    "ANALYTICAL_PLANNER_PROMPT_VERSION"
)


EXPECTED_PROMPT_VERSION = (
    "analytical_planner_prompt_v0.9_baseline"
)


# ============================================================
# SAFE AST STRING EVALUATION
# ============================================================

def _safe_eval_string_expression(
    node: ast.AST,
) -> str:
    """
    Resolve only a deliberately tiny subset of Python
    expressions that can safely construct a static string.

    Supported forms include:
    - a string literal;
    - static string concatenation with +;
    - zero-argument strip(), lstrip() or rstrip() applied
      to another supported static string expression.

    No arbitrary Python code is executed.
    """

    # ========================================================
    # STRING LITERAL
    # ========================================================

    if isinstance(
        node,
        ast.Constant,
    ):

        if isinstance(
            node.value,
            str,
        ):

            return (
                node.value
            )


        raise ValueError(
            "Expected a string literal, got "
            f"{type(node.value).__name__}."
        )


    # ========================================================
    # STATIC STRING CONCATENATION
    # ========================================================

    if (
        isinstance(
            node,
            ast.BinOp,
        )
        and isinstance(
            node.op,
            ast.Add,
        )
    ):

        left = (
            _safe_eval_string_expression(
                node.left
            )
        )


        right = (
            _safe_eval_string_expression(
                node.right
            )
        )


        return (
            left
            + right
        )


    # ========================================================
    # SAFE STRING METHODS
    # ========================================================

    if isinstance(
        node,
        ast.Call,
    ):

        if not isinstance(
            node.func,
            ast.Attribute,
        ):

            raise ValueError(
                "Unsupported function call in static "
                "string expression."
            )


        method_name = (
            node.func.attr
        )


        if (
            method_name
            not in {
                "strip",
                "lstrip",
                "rstrip",
            }
        ):

            raise ValueError(
                "Unsupported string method in static "
                "prompt expression: "
                f"{method_name}"
            )


        if (
            node.args
            or node.keywords
        ):

            raise ValueError(
                "Static prompt string methods must have "
                "no arguments."
            )


        value = (
            _safe_eval_string_expression(
                node.func.value
            )
        )


        if (
            method_name
            == "strip"
        ):

            return (
                value.strip()
            )


        if (
            method_name
            == "lstrip"
        ):

            return (
                value.lstrip()
            )


        return (
            value.rstrip()
        )


    # ========================================================
    # EVERYTHING ELSE IS REJECTED
    # ========================================================

    raise ValueError(
        "Unsupported AST expression while extracting "
        "static planner prompt: "
        f"{ast.dump(node, include_attributes=False)}"
    )


# ============================================================
# CONSTANT EXTRACTION
# ============================================================

def extract_string_constant(
    *,
    source: str,
    name: str,
) -> str:
    """
    Extract one module-level static string assignment.

    The source module is parsed but never imported or
    executed.
    """

    tree = (
        ast.parse(
            source,
            filename=str(
                SOURCE_PATH
            ),
        )
    )


    for node in (
        tree.body
    ):

        if not isinstance(
            node,
            (
                ast.Assign,
                ast.AnnAssign,
            ),
        ):

            continue


        if isinstance(
            node,
            ast.Assign,
        ):

            targets = (
                node.targets
            )


            value_node = (
                node.value
            )


        else:

            targets = [
                node.target
            ]


            value_node = (
                node.value
            )


            if (
                value_node
                is None
            ):

                continue


        for target in (
            targets
        ):

            if not isinstance(
                target,
                ast.Name,
            ):

                continue


            if (
                target.id
                != name
            ):

                continue


            value = (
                _safe_eval_string_expression(
                    value_node
                )
            )


            if not isinstance(
                value,
                str,
            ):

                raise TypeError(
                    f"{name} did not resolve to a string."
                )


            return (
                value
            )


    raise ValueError(
        "Could not find static string constant "
        f"{name!r} in {SOURCE_PATH}"
    )


# ============================================================
# TARGET CONTENT
# ============================================================

def build_target_content(
    *,
    prompt_version: str,
    system_prompt: str,
) -> str:
    """
    Create a standalone production prompt module.

    repr() preserves the exact string value without
    reconstructing the historical quoting style.
    """

    lines = [
        '"""',
        "Production prompt for the DataLens Analytical Planner v1.",
        "",
        "This prompt was promoted mechanically from the",
        "development-selected planner prompt.",
        "",
        "Runtime production code has no dependency on the",
        "evaluation package.",
        '"""',
        "",
        "",
        "# ============================================================",
        "# VERSION",
        "# ============================================================",
        "",
        (
            "ANALYTICAL_PLANNER_PROMPT_VERSION = "
            f"{prompt_version!r}"
        ),
        "",
        "",
        "# ============================================================",
        "# SYSTEM PROMPT",
        "# ============================================================",
        "",
        (
            "ANALYTICAL_PLANNER_SYSTEM_PROMPT = "
            f"{system_prompt!r}"
        ),
        "",
    ]


    return (
        "\n".join(
            lines
        )
    )


# ============================================================
# TARGET VERIFICATION
# ============================================================

def verify_target_content(
    *,
    content: str,
    prompt_version: str,
    system_prompt: str,
) -> None:
    """
    Validate the generated module before writing it.
    """

    if (
        "app.evals"
        in content
    ):

        raise ValueError(
            "Generated production planner prompt contains "
            "an evaluation-package dependency."
        )


    compile(
        content,
        str(
            TARGET_PATH
        ),
        "exec",
    )


    target_tree = (
        ast.parse(
            content,
            filename=str(
                TARGET_PATH
            ),
        )
    )


    extracted: dict[
        str,
        str,
    ] = {}


    for node in (
        target_tree.body
    ):

        if not isinstance(
            node,
            ast.Assign,
        ):

            continue


        if (
            len(
                node.targets
            )
            != 1
        ):

            continue


        target = (
            node.targets[
                0
            ]
        )


        if not isinstance(
            target,
            ast.Name,
        ):

            continue


        if (
            target.id
            not in {
                "ANALYTICAL_PLANNER_PROMPT_VERSION",
                "ANALYTICAL_PLANNER_SYSTEM_PROMPT",
            }
        ):

            continue


        value = (
            ast.literal_eval(
                node.value
            )
        )


        extracted[
            target.id
        ] = (
            value
        )


    if (
        extracted.get(
            "ANALYTICAL_PLANNER_PROMPT_VERSION"
        )
        != prompt_version
    ):

        raise ValueError(
            "Generated production prompt version differs "
            "from the historical value."
        )


    if (
        extracted.get(
            "ANALYTICAL_PLANNER_SYSTEM_PROMPT"
        )
        != system_prompt
    ):

        raise ValueError(
            "Generated production system prompt differs "
            "from the historical prompt value."
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=== DATALENS ANALYTICAL PLANNER PROMPT PROMOTION v1.0 ==="
    )


    print(
        "Promotion:",
        PROMOTION_VERSION,
    )


    print()


    if not (
        SOURCE_PATH.exists()
    ):

        raise FileNotFoundError(
            "Historical analytical planner model runner "
            "was not found:\n"
            f"{SOURCE_PATH}"
        )


    if (
        TARGET_PATH.exists()
    ):

        raise FileExistsError(
            "Production analytical planner prompt already "
            "exists. Refusing to overwrite it:\n"
            f"{TARGET_PATH}"
        )


    source = (
        SOURCE_PATH.read_text(
            encoding="utf-8",
        )
    )


    prompt_version = (
        extract_string_constant(
            source=(
                source
            ),

            name=(
                SOURCE_VERSION_NAME
            ),
        )
    )


    system_prompt = (
        extract_string_constant(
            source=(
                source
            ),

            name=(
                SOURCE_PROMPT_NAME
            ),
        )
    )


    if (
        prompt_version
        != EXPECTED_PROMPT_VERSION
    ):

        raise ValueError(
            "Unexpected historical analytical planner "
            "prompt version.\n"
            f"Expected: {EXPECTED_PROMPT_VERSION}\n"
            f"Actual:   {prompt_version}"
        )


    required_fragments = [
        "Tu es l'Analytical Planner de DataLens.",
        "join_datasets",
        "qualified_name",
        "allowed_analytical_tools",
    ]


    missing_fragments = [
        fragment

        for fragment
        in required_fragments

        if (
            fragment
            not in system_prompt
        )
    ]


    if missing_fragments:

        raise ValueError(
            "Historical analytical planner prompt is "
            "missing required locked content: "
            f"{missing_fragments}"
        )


    if (
        len(
            system_prompt
        )
        < 1000
    ):

        raise ValueError(
            "Historical planner prompt is unexpectedly "
            "short and may have been extracted incorrectly."
        )


    target_content = (
        build_target_content(
            prompt_version=(
                prompt_version
            ),

            system_prompt=(
                system_prompt
            ),
        )
    )


    verify_target_content(
        content=(
            target_content
        ),

        prompt_version=(
            prompt_version
        ),

        system_prompt=(
            system_prompt
        ),
    )


    TARGET_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    TARGET_PATH.write_text(
        target_content,
        encoding="utf-8",
    )


    print(
        "Source:",
        SOURCE_PATH,
    )


    print(
        "Target:",
        TARGET_PATH,
    )


    print(
        "Prompt version:",
        prompt_version,
    )


    print(
        "Prompt characters:",
        len(
            system_prompt
        ),
    )


    print()


    print(
        "Historical prompt value parity: PASS"
    )


    print(
        "Generated Python compile check: PASS"
    )


    print(
        "Runtime evaluation-package dependency: 0"
    )


    print()


    print(
        "Analytical Planner prompt promotion v1.0: COMPLETE"
    )


if __name__ == "__main__":
    main()
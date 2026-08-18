from __future__ import annotations

import ast

from pathlib import Path


# ============================================================
# VERSION
# ============================================================

PROMOTION_VERSION = (
    "dataset_dependency_prompt_production_promotion_v1.0"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = (
    Path(
        __file__
    )
    .resolve()
    .parent
)


SOURCE_PATH = (
    BASE_DIR
    / "app"
    / "evals"
    / "dataset_dependency_extractor_v0_8.py"
)


TARGET_PATH = (
    BASE_DIR
    / "app"
    / "ai"
    / "dataset_dependency_prompt_v1.py"
)


# ============================================================
# HISTORICAL CONSTANT NAMES
# ============================================================

SOURCE_PROMPT_NAME = (
    "SYSTEM_PROMPT_V0_8"
)


SOURCE_VERSION_NAME = (
    "DATASET_DEPENDENCY_PROMPT_VERSION"
)


SOURCE_MODEL_NAME = (
    "MODEL"
)


EXPECTED_PROMPT_VERSION = (
    "dataset_dependency_prompt_v0.8_baseline"
)


EXPECTED_MODEL = (
    "qwen3:4b-instruct"
)


# ============================================================
# SAFE AST STRING EVALUATION
# ============================================================

def _safe_eval_string_expression(
    node: ast.AST,
) -> str:
    """
    Evaluate only a deliberately restricted subset of
    expressions capable of constructing a static string.

    Supported:
    - string literals;
    - static string concatenation with +;
    - zero-argument strip(), lstrip() and rstrip().

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
    # STATIC CONCATENATION
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
    # SAFE STATIC STRING METHODS
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
                "expression: "
                f"{method_name}"
            )


        if (
            node.args
            or node.keywords
        ):

            raise ValueError(
                "Static string method must not receive "
                "arguments."
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
    # UNSUPPORTED EXPRESSION
    # ========================================================

    raise ValueError(
        "Unsupported AST expression while extracting "
        "historical dependency prompt: "
        f"{ast.dump(node, include_attributes=False)}"
    )


# ============================================================
# STRING CONSTANT EXTRACTION
# ============================================================

def extract_string_constant(
    *,
    source: str,
    name: str,
) -> str:
    """
    Extract one module-level static string constant.

    The historical module is parsed but never imported or
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
# TARGET MODULE
# ============================================================

def build_target_content(
    *,
    prompt_version: str,
    model: str,
    system_prompt: str,
) -> str:
    """
    Build the autonomous production prompt module.

    repr() preserves the exact historical string value.
    """

    lines = [
        '"""',
        "Production prompt configuration for the DataLens",
        "Dataset Dependency Extractor v1.",
        "",
        "Values in this module were promoted mechanically",
        "from the development-selected historical extractor.",
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
            "DATASET_DEPENDENCY_PROMPT_VERSION = "
            f"{prompt_version!r}"
        ),
        "",
        "",
        "# ============================================================",
        "# MODEL",
        "# ============================================================",
        "",
        (
            "DATASET_DEPENDENCY_MODEL = "
            f"{model!r}"
        ),
        "",
        "",
        "# ============================================================",
        "# SYSTEM PROMPT",
        "# ============================================================",
        "",
        (
            "DATASET_DEPENDENCY_SYSTEM_PROMPT = "
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
# GENERATED MODULE VERIFICATION
# ============================================================

def verify_target_content(
    *,
    content: str,
    prompt_version: str,
    model: str,
    system_prompt: str,
) -> None:
    """
    Verify exact value parity before writing the production
    module.
    """

    if (
        "app.evals"
        in content
    ):

        raise ValueError(
            "Generated production dependency prompt "
            "contains an evaluation-package dependency."
        )


    # ========================================================
    # PYTHON COMPILE
    # ========================================================

    compile(
        content,
        str(
            TARGET_PATH
        ),
        "exec",
    )


    # ========================================================
    # PARSE GENERATED CONSTANTS BACK
    # ========================================================

    tree = (
        ast.parse(
            content,
            filename=str(
                TARGET_PATH
            ),
        )
    )


    values: dict[
        str,
        str,
    ] = {}


    expected_names = {
        "DATASET_DEPENDENCY_PROMPT_VERSION",
        "DATASET_DEPENDENCY_MODEL",
        "DATASET_DEPENDENCY_SYSTEM_PROMPT",
    }


    for node in (
        tree.body
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
            not in expected_names
        ):

            continue


        value = (
            ast.literal_eval(
                node.value
            )
        )


        if not isinstance(
            value,
            str,
        ):

            raise TypeError(
                "Generated production constant "
                f"{target.id} is not a string."
            )


        values[
            target.id
        ] = (
            value
        )


    if (
        values.get(
            "DATASET_DEPENDENCY_PROMPT_VERSION"
        )
        != prompt_version
    ):

        raise ValueError(
            "Generated dependency prompt version differs "
            "from the historical value."
        )


    if (
        values.get(
            "DATASET_DEPENDENCY_MODEL"
        )
        != model
    ):

        raise ValueError(
            "Generated dependency model differs from "
            "the historical value."
        )


    if (
        values.get(
            "DATASET_DEPENDENCY_SYSTEM_PROMPT"
        )
        != system_prompt
    ):

        raise ValueError(
            "Generated dependency system prompt differs "
            "from the historical value."
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=== DATALENS DATASET DEPENDENCY PROMPT "
        "PROMOTION v1.0 ==="
    )


    print(
        "Promotion:",
        PROMOTION_VERSION,
    )


    print()


    # ========================================================
    # SOURCE GUARD
    # ========================================================

    if not (
        SOURCE_PATH.exists()
    ):

        raise FileNotFoundError(
            "Historical Dataset Dependency Extractor "
            "was not found:\n"
            f"{SOURCE_PATH}"
        )


    # ========================================================
    # NO SILENT OVERWRITE
    # ========================================================

    if (
        TARGET_PATH.exists()
    ):

        raise FileExistsError(
            "Production Dataset Dependency prompt already "
            "exists. Refusing to overwrite it:\n"
            f"{TARGET_PATH}"
        )


    source = (
        SOURCE_PATH.read_text(
            encoding="utf-8",
        )
    )


    # ========================================================
    # EXTRACT HISTORICAL VALUES
    # ========================================================

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


    model = (
        extract_string_constant(
            source=(
                source
            ),

            name=(
                SOURCE_MODEL_NAME
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


    # ========================================================
    # LOCK DEVELOPMENT-SELECTED CONFIGURATION
    # ========================================================

    if (
        prompt_version
        != EXPECTED_PROMPT_VERSION
    ):

        raise ValueError(
            "Unexpected historical dependency prompt "
            "version.\n"
            f"Expected: {EXPECTED_PROMPT_VERSION}\n"
            f"Actual:   {prompt_version}"
        )


    if (
        model
        != EXPECTED_MODEL
    ):

        raise ValueError(
            "Unexpected historical dependency model.\n"
            f"Expected: {EXPECTED_MODEL}\n"
            f"Actual:   {model}"
        )


    # ========================================================
    # CONTENT GUARDS
    # ========================================================

    required_fragments = [
        (
            "Tu es le Dataset Dependency "
            "Extractor de DataLens."
        ),
        "ANALYTICAL REQUIREMENT",
        "déterminer si une jointure est possible",
        "dataset_id",
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


    if (
        missing_fragments
    ):

        raise ValueError(
            "Historical dependency prompt is missing "
            "expected locked content: "
            f"{missing_fragments}"
        )


    if (
        len(
            system_prompt
        )
        < 1000
    ):

        raise ValueError(
            "Historical dependency prompt is unexpectedly "
            "short and may have been extracted incorrectly."
        )


    # ========================================================
    # BUILD TARGET
    # ========================================================

    target_content = (
        build_target_content(
            prompt_version=(
                prompt_version
            ),

            model=(
                model
            ),

            system_prompt=(
                system_prompt
            ),
        )
    )


    # ========================================================
    # VERIFY BEFORE WRITE
    # ========================================================

    verify_target_content(
        content=(
            target_content
        ),

        prompt_version=(
            prompt_version
        ),

        model=(
            model
        ),

        system_prompt=(
            system_prompt
        ),
    )


    # ========================================================
    # WRITE
    # ========================================================

    TARGET_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    TARGET_PATH.write_text(
        target_content,
        encoding="utf-8",
    )


    # ========================================================
    # REPORT
    # ========================================================

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
        "Model:",
        model,
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
        "Historical model value parity: PASS"
    )


    print(
        "Generated Python compile check: PASS"
    )


    print(
        "Runtime evaluation-package dependency: 0"
    )


    print()


    print(
        "Dataset Dependency prompt promotion v1.0: COMPLETE"
    )


if __name__ == "__main__":
    main()
from __future__ import annotations


from typing import (
    Literal,
)


from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


from app.adaptation.contracts import (
    BaseModelReference,
    QLORA_TARGET_RESOLVER_RULE_VERSION,
    QLoRAParameters,
)


# ============================================================
# TYPES
# ============================================================


SupportedTargetModelFamily = Literal[
    "gemma3",
]


# ============================================================
# FORBIDDEN NAMESPACES
# ============================================================


FORBIDDEN_TARGET_NAMESPACE_SEGMENTS = frozenset(
    {
        "vision_tower",
        "vision_model",
        "image_encoder",
        "multi_modal_projector",
        "multimodal_projector",
    }
)


FORBIDDEN_LANGUAGE_OUTPUT_MODULES = frozenset(
    {
        "lm_head",
        "output_layer",
    }
)


# ============================================================
# RESOLUTION RESULT
# ============================================================


class QLoRATargetResolution(
    BaseModel
):
    """
    Server-owned, immutable resolution of the concrete modules
    that may receive LoRA adapters.

    target_modules contains fully-qualified module names.

    The resolver deliberately returns complete paths rather than
    generic suffixes such as "q_proj". This prevents PEFT from
    matching identically named projections outside the language
    model, including a multimodal vision tower.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    model_family: SupportedTargetModelFamily


    language_model_root: str = Field(
        min_length=1,
    )


    target_modules: tuple[
        str,
        ...,
    ] = Field(
        min_length=1,
    )


    target_module_count: int = Field(
        ge=1,
    )


    rule_version: Literal[
        "qlora_target_resolver_v0.1"
    ] = QLORA_TARGET_RESOLVER_RULE_VERSION


    @model_validator(
        mode="after",
    )
    def validate_resolution(
        self,
    ) -> "QLoRATargetResolution":
        if (
            self.target_module_count
            !=
            len(
                self.target_modules
            )
        ):
            raise ValueError(
                "Target module count does not match "
                "the resolved target module list."
            )


        if (
            len(
                self.target_modules
            )
            !=
            len(
                set(
                    self.target_modules
                )
            )
        ):
            raise ValueError(
                "Resolved target modules must be unique."
            )


        if (
            tuple(
                sorted(
                    self.target_modules
                )
            )
            !=
            self.target_modules
        ):
            raise ValueError(
                "Resolved target modules must be deterministically sorted."
            )


        required_prefix = (
            self.language_model_root
            +
            "."
        )


        invalid_roots = [
            target

            for target
            in self.target_modules

            if not target.startswith(
                required_prefix
            )
        ]


        if invalid_roots:
            raise ValueError(
                "Resolved target modules escaped the language "
                f"model namespace: {invalid_roots}."
            )


        forbidden_targets = [
            target

            for target
            in self.target_modules

            if any(
                segment
                in
                FORBIDDEN_TARGET_NAMESPACE_SEGMENTS

                for segment
                in target.split(
                    "."
                )
            )
        ]


        if forbidden_targets:
            raise ValueError(
                "Resolved target modules contain forbidden "
                f"multimodal namespaces: {forbidden_targets}."
            )


        return self


# ============================================================
# HELPERS
# ============================================================


def _is_supported_linear_module(
    module: object,
) -> bool:
    """
    Detect the linear layer implementations that may appear before
    or after 4-bit quantization.

    torch and bitsandbytes are imported lazily so importing
    app.adaptation contracts does not add training dependencies to
    the normal FastAPI runtime.
    """

    try:
        import torch

    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required to resolve QLoRA target modules. "
            "Use the DataLens adaptation environment."
        ) from error


    if isinstance(
        module,
        torch.nn.Linear,
    ):
        return True


    module_type = type(
        module
    )


    module_name = (
        module_type.__name__
    )


    module_package = (
        module_type.__module__
    )


    if (
        module_name
        in {
            "Linear4bit",
            "Linear8bitLt",
        }
        and
        module_package.startswith(
            "bitsandbytes"
        )
    ):
        return True


    return False


def _resolve_language_model_object(
    model: object,
) -> object:
    """
    Resolve the server-owned Gemma 3 language model object.

    Two layouts are supported because wrappers may expose the
    language model directly or below a top-level model container.

    Multiple distinct matches are rejected rather than guessed.
    """

    candidates: list[
        object,
    ] = []


    direct_language_model = getattr(
        model,
        "language_model",
        None,
    )


    if (
        direct_language_model
        is not None
    ):
        candidates.append(
            direct_language_model
        )


    model_container = getattr(
        model,
        "model",
        None,
    )


    if (
        model_container
        is not None
    ):
        nested_language_model = getattr(
            model_container,
            "language_model",
            None,
        )


        if (
            nested_language_model
            is not None
        ):
            candidates.append(
                nested_language_model
            )


    unique_candidates: dict[
        int,
        object,
    ] = {
        id(
            candidate
        ):
            candidate

        for candidate
        in candidates
    }


    if not unique_candidates:
        raise ValueError(
            "Gemma 3 language model root could not be resolved."
        )


    if (
        len(
            unique_candidates
        )
        !=
        1
    ):
        raise ValueError(
            "Multiple distinct Gemma 3 language model roots "
            "were found. Target resolution is ambiguous."
        )


    return next(
        iter(
            unique_candidates.values()
        )
    )


def _resolve_module_path(
    *,
    model: object,
    target_object: object,
) -> str:
    """
    Resolve the canonical fully-qualified path of one module object
    from the complete model namespace.
    """

    named_modules = getattr(
        model,
        "named_modules",
        None,
    )


    if (
        named_modules
        is None
        or
        not callable(
            named_modules
        )
    ):
        raise TypeError(
            "QLoRA target resolver requires a torch-compatible "
            "model exposing named_modules()."
        )


    matches = [
        name

        for (
            name,
            module,
        )
        in named_modules()

        if (
            module
            is
            target_object
        )
    ]


    if (
        len(
            matches
        )
        !=
        1
    ):
        raise ValueError(
            "Language model root must resolve to exactly one "
            f"canonical module path. Found: {matches}."
        )


    root = (
        matches[
            0
        ]
        .strip(
            "."
        )
    )


    if not root:
        raise ValueError(
            "Language model root may not be the complete "
            "multimodal model."
        )


    return root


# ============================================================
# PUBLIC RESOLVER
# ============================================================


def resolve_qlora_target_modules(
    *,
    model: object,
    base_model: BaseModelReference,
    lora: QLoRAParameters,
) -> QLoRATargetResolution:
    """
    Resolve concrete LoRA targets from the loaded server-owned
    model.

    Security / correctness invariants:

    - v0.1 supports Gemma 3 only;
    - target strategy must be language_model_all_linear;
    - only modules beneath the resolved language-model root are
      eligible;
    - multimodal namespaces remain forbidden;
    - output heads remain frozen;
    - caller-controlled target module names are never accepted;
    - result ordering is deterministic.
    """

    if (
        base_model.model_family
        !=
        "gemma3"
    ):
        raise ValueError(
            "QLoRA target resolver v0.1 supports only "
            "model_family='gemma3'."
        )


    if (
        lora.target_strategy
        !=
        "language_model_all_linear"
    ):
        raise ValueError(
            "Unsupported QLoRA target strategy."
        )


    if (
        lora.target_resolver_rule_version
        !=
        QLORA_TARGET_RESOLVER_RULE_VERSION
    ):
        raise ValueError(
            "QLoRA target resolver rule version mismatch."
        )


    language_model = (
        _resolve_language_model_object(
            model
        )
    )


    language_model_root = (
        _resolve_module_path(
            model=model,
            target_object=language_model,
        )
    )


    required_prefix = (
        language_model_root
        +
        "."
    )


    target_modules: list[
        str,
    ] = []


    named_modules = getattr(
        model,
        "named_modules",
    )


    for (
        module_name,
        module,
    ) in named_modules():
        if not module_name.startswith(
            required_prefix
        ):
            continue


        segments = (
            module_name.split(
                "."
            )
        )


        if any(
            segment
            in
            FORBIDDEN_TARGET_NAMESPACE_SEGMENTS

            for segment
            in segments
        ):
            raise ValueError(
                "A forbidden multimodal namespace appeared "
                "inside the resolved language model root: "
                f"{module_name}."
            )


        leaf_name = (
            segments[
                -1
            ]
        )


        if (
            leaf_name
            in
            FORBIDDEN_LANGUAGE_OUTPUT_MODULES
        ):
            continue


        if not _is_supported_linear_module(
            module
        ):
            continue


        target_modules.append(
            module_name
        )


    resolved_targets = tuple(
        sorted(
            set(
                target_modules
            )
        )
    )


    if not resolved_targets:
        raise ValueError(
            "No eligible language-model linear modules were "
            "resolved for QLoRA."
        )


    return QLoRATargetResolution(
        model_family="gemma3",
        language_model_root=
            language_model_root,
        target_modules=
            resolved_targets,
        target_module_count=
            len(
                resolved_targets
            ),
    )
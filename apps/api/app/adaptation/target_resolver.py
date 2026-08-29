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
    model, including multimodal vision components.
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
                "Resolved target modules must be "
                "deterministically sorted."
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
# RUNTIME MODEL HELPERS
# ============================================================


def _runtime_model_type(
    model: object,
) -> str | None:
    config = getattr(
        model,
        "config",
        None,
    )


    if config is None:
        return None


    model_type = getattr(
        config,
        "model_type",
        None,
    )


    if (
        model_type is not None
        and
        not isinstance(
            model_type,
            str,
        )
    ):
        raise TypeError(
            "Runtime model_type must be a string when present."
        )


    return model_type


def _named_modules(
    model: object,
):
    named_modules = getattr(
        model,
        "named_modules",
        None,
    )


    if (
        named_modules is None
        or
        not callable(
            named_modules
        )
    ):
        raise TypeError(
            "QLoRA target resolver requires a torch-compatible "
            "model exposing named_modules()."
        )


    return named_modules()


# ============================================================
# LINEAR DETECTION
# ============================================================


def _is_supported_linear_module(
    module: object,
) -> bool:
    """
    Detect linear layer implementations that may appear before
    or after bitsandbytes quantization.

    torch is imported lazily so importing app.adaptation contracts
    does not force training dependencies into the normal runtime.
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


# ============================================================
# LANGUAGE ROOT RESOLUTION
# ============================================================


def _resolve_multimodal_language_model_object(
    model: object,
) -> object | None:
    """
    Resolve an explicitly exposed language_model from a
    multimodal wrapper.

    Multiple distinct candidates are rejected rather than guessed.
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
        return None


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


def _resolve_language_model_object(
    model: object,
) -> object:
    """
    Resolve the server-owned Gemma 3 language-model object.

    Supported layouts:

    1. Multimodal wrapper:
       model.language_model or model.model.language_model

    2. Preferred DataLens text-only wrapper:
       Gemma3ForCausalLM.model

       This second form is accepted only when the runtime config
       explicitly identifies model_type='gemma3_text'.

    Arbitrary objects exposing only a generic .model attribute are
    therefore not trusted.
    """

    multimodal_language_model = (
        _resolve_multimodal_language_model_object(
            model
        )
    )


    if (
        multimodal_language_model
        is not None
    ):
        return multimodal_language_model


    runtime_model_type = (
        _runtime_model_type(
            model
        )
    )


    if (
        runtime_model_type
        !=
        "gemma3_text"
    ):
        raise ValueError(
            "Gemma 3 language model root could not be resolved."
        )


    text_model = getattr(
        model,
        "model",
        None,
    )


    if (
        text_model is None
    ):
        raise ValueError(
            "Gemma 3 text-only runtime declared model_type="
            "'gemma3_text' but did not expose the expected "
            "top-level model object."
        )


    return text_model


def _resolve_module_path(
    *,
    model: object,
    target_object: object,
) -> str:
    """
    Resolve the canonical fully-qualified path of one module object
    from the complete model namespace.
    """

    matches = [
        name

        for (
            name,
            module,
        )
        in _named_modules(
            model
        )

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
            "Language model root may not be the complete model."
        )


    return root


# ============================================================
# MULTIMODAL SAFETY
# ============================================================


def _validate_text_only_runtime_surface(
    model: object,
) -> None:
    """
    For Gemma3ForCausalLM, fail closed if any multimodal namespace
    unexpectedly appears anywhere in the runtime model.

    This is stronger than merely ignoring those modules.
    """

    if (
        _runtime_model_type(
            model
        )
        !=
        "gemma3_text"
    ):
        return


    forbidden_modules = [
        name

        for (
            name,
            _,
        )
        in _named_modules(
            model
        )

        if any(
            segment
            in
            FORBIDDEN_TARGET_NAMESPACE_SEGMENTS

            for segment
            in name.split(
                "."
            )
        )
    ]


    if forbidden_modules:
        raise ValueError(
            "Text-only Gemma 3 runtime unexpectedly exposes "
            "multimodal namespaces: "
            f"{forbidden_modules}."
        )


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
    - the preferred text-only Gemma3ForCausalLM runtime must
      identify itself as gemma3_text;
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
        base_model.modality_scope
        !=
        "text_only"
    ):
        raise ValueError(
            "QLoRA target resolver v0.1 requires "
            "modality_scope='text_only'."
        )


    if (
        base_model.use_multimodal_inputs
        is not False
    ):
        raise ValueError(
            "QLoRA target resolver v0.1 forbids "
            "multimodal training inputs."
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


    runtime_model_type = (
        _runtime_model_type(
            model
        )
    )


    if (
        runtime_model_type
        not in {
            None,
            "gemma3",
            "gemma3_text",
        }
    ):
        raise ValueError(
            "Unexpected runtime model type for Gemma 3 QLoRA "
            f"target resolution: {runtime_model_type}."
        )


    _validate_text_only_runtime_surface(
        model
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


    for (
        module_name,
        module,
    ) in _named_modules(
        model
    ):
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

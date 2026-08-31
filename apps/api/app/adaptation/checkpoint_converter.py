from __future__ import annotations


import gc
import hashlib
import json
import os
import shutil
import tempfile


from collections import (
    defaultdict,
)

from pathlib import (
    Path,
)

from typing import (
    Any,
)


from huggingface_hub import (
    hf_hub_download,
)

from safetensors import (
    safe_open,
)

from safetensors.torch import (
    save_file,
)

from transformers import (
    AutoConfig,
)


# ============================================================
# VERSION
# ============================================================


GEMMA3_TEXT_CHECKPOINT_CONVERTER_RULE_VERSION = (
    "gemma3_text_checkpoint_converter_v0.1"
)


# ============================================================
# CONSTANTS
# ============================================================


SOURCE_LANGUAGE_PREFIX = (
    "language_model."
)


EXPECTED_SOURCE_LANGUAGE_TENSOR_COUNT = (
    444
)


EXPECTED_DECODER_LAYER_COUNT = (
    34
)


EXPECTED_LORA_PROJECTION_COUNT = (
    238
)


OUTPUT_INDEX_FILENAME = (
    "model.safetensors.index.json"
)


OUTPUT_CONFIG_FILENAME = (
    "config.json"
)


OUTPUT_MANIFEST_FILENAME = (
    "datalens_checkpoint_manifest.json"
)


FORBIDDEN_KEY_FRAGMENTS = (
    "vision_tower",
    "vision_model",
    "multi_modal_projector",
    "multimodal_projector",
)


# ============================================================
# HASHING
# ============================================================


def _sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()


    with path.open(
        "rb",
    ) as handle:
        while True:
            chunk = handle.read(
                1024
                *
                1024
                *
                8
            )


            if not chunk:
                break


            digest.update(
                chunk
            )


    return digest.hexdigest()


# ============================================================
# JSON
# ============================================================


def _write_json(
    *,
    path: Path,
    payload: dict[
        str,
        Any,
    ],
) -> None:
    serialized = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
    )


    path.write_text(
        serialized
        +
        "\n",
        encoding="utf-8",
        newline="\n",
    )


# ============================================================
# SOURCE INDEX
# ============================================================


def _load_source_weight_map(
    *,
    source_model_id: str,
    source_revision: str,
) -> tuple[
    Path,
    dict[
        str,
        str,
    ],
]:
    index_path = Path(
        hf_hub_download(
            repo_id=source_model_id,
            filename=OUTPUT_INDEX_FILENAME,
            revision=source_revision,
        )
    )


    payload = json.loads(
        index_path.read_text(
            encoding="utf-8"
        )
    )


    weight_map = payload.get(
        "weight_map"
    )


    if not isinstance(
        weight_map,
        dict,
    ):
        raise RuntimeError(
            "Source checkpoint index does not expose "
            "a valid weight_map."
        )


    normalized_weight_map: dict[
        str,
        str,
    ] = {}


    for (
        key,
        value,
    ) in weight_map.items():
        if (
            not isinstance(
                key,
                str,
            )
            or
            not isinstance(
                value,
                str,
            )
        ):
            raise RuntimeError(
                "Source checkpoint weight_map contains "
                "non-string entries."
            )


        normalized_weight_map[
            key
        ] = value


    return (
        index_path,
        normalized_weight_map,
    )


# ============================================================
# KEY MAPPING
# ============================================================


def _build_language_key_mapping(
    *,
    weight_map: dict[
        str,
        str,
    ],
) -> dict[
    str,
    str,
]:
    source_language_keys = sorted(
        key

        for key
        in weight_map

        if key.startswith(
            SOURCE_LANGUAGE_PREFIX
        )
    )


    if (
        len(
            source_language_keys
        )
        !=
        EXPECTED_SOURCE_LANGUAGE_TENSOR_COUNT
    ):
        raise RuntimeError(
            "Expected exactly "
            f"{EXPECTED_SOURCE_LANGUAGE_TENSOR_COUNT} "
            "Gemma 3 language tensors, found "
            f"{len(source_language_keys)}."
        )


    mapping = {
        source_key:
            source_key[
                len(
                    SOURCE_LANGUAGE_PREFIX
                ):
            ]

        for source_key
        in source_language_keys
    }


    target_keys = list(
        mapping.values()
    )


    if (
        len(
            target_keys
        )
        !=
        len(
            set(
                target_keys
            )
        )
    ):
        raise RuntimeError(
            "Text checkpoint conversion produced "
            "target-key collisions."
        )


    forbidden_targets = [
        target_key

        for target_key
        in target_keys

        if any(
            fragment
            in
            target_key

            for fragment
            in FORBIDDEN_KEY_FRAGMENTS
        )
    ]


    if forbidden_targets:
        raise RuntimeError(
            "Multimodal weights escaped into the "
            "text checkpoint mapping."
        )


    required_keys = {
        "model.embed_tokens.weight",
        "model.norm.weight",
    }


    missing_required = sorted(
        required_keys
        -
        set(
            target_keys
        )
    )


    if missing_required:
        raise RuntimeError(
            "Required Gemma 3 text weights are missing: "
            f"{missing_required}."
        )


    if (
        "lm_head.weight"
        in target_keys
    ):
        raise RuntimeError(
            "Unexpected standalone lm_head.weight "
            "in source Gemma 3 checkpoint."
        )


    layer_ids = sorted(
        {
            int(
                target_key.split(
                    "."
                )[
                    2
                ]
            )

            for target_key
            in target_keys

            if target_key.startswith(
                "model.layers."
            )
        }
    )


    if (
        layer_ids
        !=
        list(
            range(
                EXPECTED_DECODER_LAYER_COUNT
            )
        )
    ):
        raise RuntimeError(
            "Unexpected Gemma 3 decoder layer surface."
        )


    projection_suffixes = (
        ".self_attn.q_proj.weight",
        ".self_attn.k_proj.weight",
        ".self_attn.v_proj.weight",
        ".self_attn.o_proj.weight",
        ".mlp.gate_proj.weight",
        ".mlp.up_proj.weight",
        ".mlp.down_proj.weight",
    )


    projection_count = sum(
        1

        for target_key
        in target_keys

        if target_key.endswith(
            projection_suffixes
        )
    )


    if (
        projection_count
        !=
        EXPECTED_LORA_PROJECTION_COUNT
    ):
        raise RuntimeError(
            "Expected exactly "
            f"{EXPECTED_LORA_PROJECTION_COUNT} "
            "LoRA-compatible projection weights, found "
            f"{projection_count}."
        )


    return mapping


# ============================================================
# TEXT CONFIG
# ============================================================


def _build_text_config_payload(
    *,
    source_model_id: str,
    source_revision: str,
) -> dict[
    str,
    Any,
]:
    full_config = AutoConfig.from_pretrained(
        source_model_id,
        revision=source_revision,
        trust_remote_code=False,
    )


    text_config = getattr(
        full_config,
        "text_config",
        None,
    )


    if text_config is None:
        raise RuntimeError(
            "Source Gemma 3 configuration does not "
            "expose text_config."
        )


    if (
        getattr(
            text_config,
            "model_type",
            None,
        )
        !=
        "gemma3_text"
    ):
        raise RuntimeError(
            "Unexpected Gemma 3 text model type."
        )


    if not getattr(
        text_config,
        "tie_word_embeddings",
        False,
    ):
        raise RuntimeError(
            "Gemma 3 text configuration must tie "
            "input and output embeddings."
        )


    payload = (
        text_config.to_dict()
    )


    payload[
        "architectures"
    ] = [
        "Gemma3ForCausalLM",
    ]


    payload[
        "_name_or_path"
    ] = (
        source_model_id
    )


    return payload


# ============================================================
# OUTPUT VALIDATION
# ============================================================


def _validate_output_checkpoint(
    *,
    output_dir: Path,
    expected_target_keys: set[
        str,
    ],
) -> None:
    index_path = (
        output_dir
        /
        OUTPUT_INDEX_FILENAME
    )


    payload = json.loads(
        index_path.read_text(
            encoding="utf-8"
        )
    )


    weight_map = payload.get(
        "weight_map"
    )


    if not isinstance(
        weight_map,
        dict,
    ):
        raise RuntimeError(
            "Converted checkpoint index is invalid."
        )


    indexed_keys = set(
        weight_map
    )


    if (
        indexed_keys
        !=
        expected_target_keys
    ):
        missing = sorted(
            expected_target_keys
            -
            indexed_keys
        )


        unexpected = sorted(
            indexed_keys
            -
            expected_target_keys
        )


        raise RuntimeError(
            "Converted checkpoint index mismatch. "
            f"Missing={missing}; "
            f"Unexpected={unexpected}."
        )


    actual_keys: set[
        str,
    ] = set()


    shard_names = sorted(
        set(
            weight_map.values()
        )
    )


    for shard_name in shard_names:
        shard_path = (
            output_dir
            /
            shard_name
        )


        if not shard_path.is_file():
            raise RuntimeError(
                "Converted checkpoint shard is missing: "
                f"{shard_name}."
            )


        with safe_open(
            shard_path,
            framework="pt",
            device="cpu",
        ) as handle:
            shard_keys = set(
                handle.keys()
            )


        overlap = (
            actual_keys
            &
            shard_keys
        )


        if overlap:
            raise RuntimeError(
                "Converted checkpoint contains duplicate "
                f"tensor keys across shards: {sorted(overlap)}."
            )


        actual_keys.update(
            shard_keys
        )


    if (
        actual_keys
        !=
        expected_target_keys
    ):
        raise RuntimeError(
            "Converted safetensors contents do not match "
            "the expected text-model key surface."
        )


    config = AutoConfig.from_pretrained(
        output_dir,
        trust_remote_code=False,
        local_files_only=True,
    )


    if (
        config.model_type
        !=
        "gemma3_text"
    ):
        raise RuntimeError(
            "Converted checkpoint config does not identify "
            "Gemma 3 text."
        )


    architectures = (
        config.architectures
        or
        []
    )


    if (
        "Gemma3ForCausalLM"
        not in architectures
    ):
        raise RuntimeError(
            "Converted checkpoint config does not declare "
            "Gemma3ForCausalLM."
        )


    if not getattr(
        config,
        "tie_word_embeddings",
        False,
    ):
        raise RuntimeError(
            "Converted checkpoint lost tied-embedding policy."
        )


# ============================================================
# PUBLIC CONVERTER
# ============================================================


def convert_gemma3_text_checkpoint(
    *,
    source_model_id: str,
    source_revision: str,
    output_dir: Path,
) -> Path:
    """
    Create a deterministic text-only Gemma 3 checkpoint from the
    pinned official multimodal checkpoint.

    v0.1 invariants:

    - source revision must be an immutable 40-character Git SHA;
    - only language_model.* tensors may be copied;
    - the language_model. prefix is removed;
    - vision and multimodal projector tensors are forbidden;
    - lm_head.weight is not synthesized because Gemma 3 ties the
      output head to model.embed_tokens.weight;
    - output is written to a temporary directory and published only
      after structural validation succeeds;
    - an existing destination is never overwritten.
    """

    if (
        len(
            source_revision
        )
        !=
        40
        or
        any(
            character
            not in
            "0123456789abcdef"

            for character
            in source_revision
        )
    ):
        raise ValueError(
            "source_revision must be an immutable "
            "lowercase 40-character Git SHA."
        )


    output_dir = (
        output_dir
        .expanduser()
        .resolve()
    )


    if output_dir.exists():
        raise FileExistsError(
            "Converted checkpoint destination already exists: "
            f"{output_dir}"
        )


    output_dir.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    (
        source_index_path,
        source_weight_map,
    ) = _load_source_weight_map(
        source_model_id=source_model_id,
        source_revision=source_revision,
    )


    mapping = (
        _build_language_key_mapping(
            weight_map=source_weight_map,
        )
    )


    expected_target_keys = set(
        mapping.values()
    )


    text_config_payload = (
        _build_text_config_payload(
            source_model_id=source_model_id,
            source_revision=source_revision,
        )
    )


    source_keys_by_shard: dict[
        str,
        list[
            str,
        ],
    ] = defaultdict(
        list
    )


    for source_key in sorted(
        mapping
    ):
        source_keys_by_shard[
            source_weight_map[
                source_key
            ]
        ].append(
            source_key
        )


    temporary_dir = Path(
        tempfile.mkdtemp(
            prefix=(
                output_dir.name
                +
                ".tmp-"
            ),
            dir=str(
                output_dir.parent
            ),
        )
    )


    source_shard_manifest: list[
        dict[
            str,
            Any,
        ]
    ] = []


    output_shard_manifest: list[
        dict[
            str,
            Any,
        ]
    ] = []


    target_weight_map: dict[
        str,
        str,
    ] = {}


    total_tensor_bytes = 0


    try:
        _write_json(
            path=(
                temporary_dir
                /
                OUTPUT_CONFIG_FILENAME
            ),
            payload=text_config_payload,
        )


        for shard_name in sorted(
            source_keys_by_shard
        ):
            source_path = Path(
                hf_hub_download(
                    repo_id=source_model_id,
                    filename=shard_name,
                    revision=source_revision,
                )
            )


            selected_source_keys = (
                source_keys_by_shard[
                    shard_name
                ]
            )


            tensors: dict[
                str,
                Any,
            ] = {}


            shard_tensor_bytes = 0


            with safe_open(
                source_path,
                framework="pt",
                device="cpu",
            ) as handle:
                available_keys = set(
                    handle.keys()
                )


                missing_keys = [
                    key

                    for key
                    in selected_source_keys

                    if key
                    not in
                    available_keys
                ]


                if missing_keys:
                    raise RuntimeError(
                        "Source shard does not contain indexed "
                        f"language tensors: {missing_keys}."
                    )


                for source_key in selected_source_keys:
                    target_key = (
                        mapping[
                            source_key
                        ]
                    )


                    tensor = (
                        handle.get_tensor(
                            source_key
                        )
                    )


                    tensors[
                        target_key
                    ] = tensor


                    tensor_bytes = (
                        tensor.numel()
                        *
                        tensor.element_size()
                    )


                    shard_tensor_bytes += (
                        tensor_bytes
                    )


                    target_weight_map[
                        target_key
                    ] = shard_name


                output_shard_path = (
                    temporary_dir
                    /
                    shard_name
                )


                save_file(
                    tensors,
                    output_shard_path,
                    metadata={
                        "format":
                            "pt",
                        "datalens_rule_version":
                            GEMMA3_TEXT_CHECKPOINT_CONVERTER_RULE_VERSION,
                    },
                )


            total_tensor_bytes += (
                shard_tensor_bytes
            )


            source_shard_manifest.append(
                {
                    "filename":
                        shard_name,
                    "sha256":
                        _sha256_file(
                            source_path
                        ),
                    "size_bytes":
                        source_path.stat().st_size,
                    "selected_language_tensor_count":
                        len(
                            selected_source_keys
                        ),
                }
            )


            output_shard_path = (
                temporary_dir
                /
                shard_name
            )


            output_shard_manifest.append(
                {
                    "filename":
                        shard_name,
                    "sha256":
                        _sha256_file(
                            output_shard_path
                        ),
                    "size_bytes":
                        output_shard_path.stat().st_size,
                    "tensor_count":
                        len(
                            tensors
                        ),
                }
            )


            del tensors

            gc.collect()


        output_index_payload = {
            "metadata": {
                "total_size":
                    total_tensor_bytes,
            },
            "weight_map": {
                key:
                    target_weight_map[
                        key
                    ]

                for key
                in sorted(
                    target_weight_map
                )
            },
        }


        output_index_path = (
            temporary_dir
            /
            OUTPUT_INDEX_FILENAME
        )


        _write_json(
            path=output_index_path,
            payload=output_index_payload,
        )


        _validate_output_checkpoint(
            output_dir=temporary_dir,
            expected_target_keys=
                expected_target_keys,
        )


        source_index_sha256 = (
            _sha256_file(
                source_index_path
            )
        )


        config_path = (
            temporary_dir
            /
            OUTPUT_CONFIG_FILENAME
        )


        manifest_payload = {
            "converter_rule_version":
                GEMMA3_TEXT_CHECKPOINT_CONVERTER_RULE_VERSION,
            "source": {
                "model_id":
                    source_model_id,
                "revision":
                    source_revision,
                "index_filename":
                    OUTPUT_INDEX_FILENAME,
                "index_sha256":
                    source_index_sha256,
                "language_prefix":
                    SOURCE_LANGUAGE_PREFIX,
                "language_tensor_count":
                    len(
                        mapping
                    ),
                "shards":
                    source_shard_manifest,
            },
            "target": {
                "model_type":
                    "gemma3_text",
                "architecture":
                    "Gemma3ForCausalLM",
                "tensor_count":
                    len(
                        expected_target_keys
                    ),
                "decoder_layer_count":
                    EXPECTED_DECODER_LAYER_COUNT,
                "lora_projection_weight_count":
                    EXPECTED_LORA_PROJECTION_COUNT,
                "vision_tensor_count":
                    0,
                "projector_tensor_count":
                    0,
                "standalone_lm_head_weight":
                    False,
                "tie_word_embeddings":
                    True,
                "config_sha256":
                    _sha256_file(
                        config_path
                    ),
                "index_sha256":
                    _sha256_file(
                        output_index_path
                    ),
                "shards":
                    output_shard_manifest,
            },
        }


        manifest_path = (
            temporary_dir
            /
            OUTPUT_MANIFEST_FILENAME
        )


        _write_json(
            path=manifest_path,
            payload=manifest_payload,
        )


        os.replace(
            temporary_dir,
            output_dir,
        )


        return (
            output_dir
            /
            OUTPUT_MANIFEST_FILENAME
        )


    except Exception:
        shutil.rmtree(
            temporary_dir,
            ignore_errors=True,
        )


        raise

from __future__ import annotations

import math

from app.adaptation.training_runner import (
    EXPECTED_EXAMPLE_COUNT,
    EXPECTED_TOTAL_MICRO_BATCHES,
    EXPECTED_TOTAL_OPTIMIZER_STEPS,
    RUNNER_RULE_VERSION,
    accumulation_groups,
    deterministic_epoch_order,
    supervised_token_loss_scales,
    validate_static_contract,
    weighted_group_loss,
)


print(
    "=== DATALENS TRAINING RUNNER v0.1 ==="
)

print()


# ============================================================
# 1. RUNNER IDENTITY
# ============================================================


if (
    RUNNER_RULE_VERSION
    !=
    "qlora_training_runner_v0.1"
):
    raise RuntimeError(
        "Unexpected runner rule version."
    )


print(
    "Runner rule version: PASS"
)


# ============================================================
# 2. STATIC CONTRACT VALIDATION
#
# This validation may import ordinary Python modules from the
# adaptation package. Its safety boundary is NOT "torch absent
# from sys.modules".
#
# The real invariant is that no model, optimizer, backward pass,
# optimizer step, training output, or Final Acceptance execution
# occurs during this test.
# ============================================================


static = validate_static_contract()


manifest = static[
    "manifest"
]


if (
    manifest[
        "dataset"
    ][
        "example_count"
    ]
    !=
    EXPECTED_EXAMPLE_COUNT
):
    raise RuntimeError(
        "Unexpected training example count."
    )


if (
    manifest[
        "training"
    ][
        "total_micro_batches"
    ]
    !=
    EXPECTED_TOTAL_MICRO_BATCHES
):
    raise RuntimeError(
        "Unexpected micro-batch plan."
    )


if (
    manifest[
        "training"
    ][
        "total_optimizer_steps"
    ]
    !=
    EXPECTED_TOTAL_OPTIMIZER_STEPS
):
    raise RuntimeError(
        "Unexpected optimizer-step plan."
    )


if (
    manifest[
        "authorization"
    ][
        "optimizer_step_authorized_at_manifest_creation"
    ]
    is not False
):
    raise RuntimeError(
        "Manifest prematurely authorizes "
        "optimizer.step()."
    )


if (
    manifest[
        "authorization"
    ][
        "final_acceptance_access_authorized"
    ]
    is not False
):
    raise RuntimeError(
        "Final Acceptance is unexpectedly "
        "authorized."
    )


print(
    "Frozen training manifest: PASS"
)


# ============================================================
# 3. DETERMINISTIC EPOCH SHUFFLE
# ============================================================


order_epoch_0 = (
    deterministic_epoch_order(
        example_count=
            EXPECTED_EXAMPLE_COUNT,
        seed=42,
        zero_based_epoch=0,
    )
)


order_epoch_0_again = (
    deterministic_epoch_order(
        example_count=
            EXPECTED_EXAMPLE_COUNT,
        seed=42,
        zero_based_epoch=0,
    )
)


order_epoch_1 = (
    deterministic_epoch_order(
        example_count=
            EXPECTED_EXAMPLE_COUNT,
        seed=42,
        zero_based_epoch=1,
    )
)


if (
    order_epoch_0
    !=
    order_epoch_0_again
):
    raise RuntimeError(
        "Epoch shuffle is not deterministic."
    )


expected_indices = list(
    range(
        EXPECTED_EXAMPLE_COUNT
    )
)


if (
    sorted(
        order_epoch_0
    )
    !=
    expected_indices
):
    raise RuntimeError(
        "Epoch 1 shuffle is not a permutation."
    )


if (
    sorted(
        order_epoch_1
    )
    !=
    expected_indices
):
    raise RuntimeError(
        "Epoch 2 shuffle is not a permutation."
    )


if (
    order_epoch_0
    ==
    order_epoch_1
):
    raise RuntimeError(
        "Two deterministic epochs unexpectedly "
        "have identical order."
    )


print(
    "Deterministic epoch shuffle: PASS"
)


# ============================================================
# 4. EXACT ACCUMULATION BOUNDARIES
# ============================================================


groups = accumulation_groups(
    order=
        order_epoch_0,
    accumulation_steps=8,
)


if (
    len(
        groups
    )
    !=
    5
):
    raise RuntimeError(
        "Expected 5 accumulation groups."
    )


if any(
    len(
        group
    )
    !=
    8
    for group in groups
):
    raise RuntimeError(
        "Expected 8 examples per "
        "accumulation group."
    )


flattened = [
    index
    for group in groups
    for index in group
]


if (
    flattened
    !=
    order_epoch_0
):
    raise RuntimeError(
        "Accumulation grouping changed "
        "epoch order."
    )


print(
    "Exact accumulation boundaries: PASS"
)


# ============================================================
# 5. TOKEN-WEIGHTED OBJECTIVE
# ============================================================


token_counts = [
    2,
    6,
]


scales = supervised_token_loss_scales(
    token_counts
)


if not math.isclose(
    scales[
        0
    ],
    0.25,
    abs_tol=1e-12,
):
    raise RuntimeError(
        "Unexpected first token-loss scale."
    )


if not math.isclose(
    scales[
        1
    ],
    0.75,
    abs_tol=1e-12,
):
    raise RuntimeError(
        "Unexpected second token-loss scale."
    )


if not math.isclose(
    sum(
        scales
    ),
    1.0,
    abs_tol=1e-12,
):
    raise RuntimeError(
        "Token-loss scales do not sum to 1."
    )


effective_loss = weighted_group_loss(
    losses=[
        1.0,
        2.0,
    ],
    token_counts=
        token_counts,
)


if not math.isclose(
    effective_loss,
    1.75,
    abs_tol=1e-12,
):
    raise RuntimeError(
        "Token-weighted group loss is incorrect."
    )


print(
    "Supervised-token weighted objective: PASS"
)


# ============================================================
# 6. FAIL-CLOSED EDGE CASES
# ============================================================


try:
    accumulation_groups(
        order=[
            0,
            1,
            2,
        ],
        accumulation_steps=2,
    )

except ValueError:
    pass

else:
    raise RuntimeError(
        "Incomplete accumulation boundary "
        "was not rejected."
    )


print(
    "Incomplete accumulation rejected: PASS"
)


try:
    supervised_token_loss_scales(
        [
            5,
            0,
        ]
    )

except ValueError:
    pass

else:
    raise RuntimeError(
        "Zero-supervision micro-batch "
        "was not rejected."
    )


print(
    "Zero-supervision batch rejected: PASS"
)


# ============================================================
# 7. STATIC SAFETY CONTRACT
# ============================================================


if (
    manifest[
        "evaluation_policy"
    ][
        "evaluation_during_training"
    ]
    is not False
):
    raise RuntimeError(
        "Evaluation-during-training policy changed."
    )


if (
    manifest[
        "evaluation_policy"
    ][
        "training_loss_is_acceptance_evidence"
    ]
    is not False
):
    raise RuntimeError(
        "Training loss became acceptance evidence."
    )


if (
    manifest[
        "final_acceptance"
    ][
        "cases_may_be_loaded_during_training"
    ]
    is not False
):
    raise RuntimeError(
        "Final Acceptance loading policy changed."
    )


print()

print(
    "TRAINING PLAN"
)

print(
    "  Examples: 40"
)

print(
    "  Epochs: 2"
)

print(
    "  Micro-batches: 80"
)

print(
    "  Accumulation groups: 10"
)

print(
    "  Planned optimizer steps: 10"
)

print(
    "  Objective: supervised-token weighted"
)


print()

print(
    "STATIC SAFETY"
)

print(
    "  Model instantiated by test: False"
)

print(
    "  CUDA operation requested by test: False"
)

print(
    "  Optimizer instantiated by test: False"
)

print(
    "  Forward requested by test: False"
)

print(
    "  Backward requested by test: False"
)

print(
    "  optimizer.step() requested by test: False"
)

print(
    "  Training output written by test: False"
)

print(
    "  Training executed: False"
)

print(
    "  Final Acceptance execution requested: False"
)


print()

print(
    "DATALENS TRAINING RUNNER v0.1: PASS"
)

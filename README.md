# LeRobot Latent SmolVLA Policy

Installable third-party LeRobot policy package that registers
`policy.type=latent_smolvla`.

This package is a standalone SmolVLA variant with optional latent supervision.
It keeps the SmolVLA-style action head and adds latent supervision modes for
mixed action/latent training without requiring online LAM inference.

The installable package name is `lerobot_policy_latent_smolvla`.

## What It Adds

- an auxiliary latent head on top of the SmolVLA backbone
- `training_mode` options: `action`, `latent`, or `multitask`
- `latent_head_mode` options: `index_cross_entropy` or `vector_diffusion`
- optional per-sample supervision masks via `latent_supervision_key` and `action_supervision_key`
- latent target routing through `latent_label_key` with default `latent_labels`

## Install

```bash
conda run -n lerobot pip install -e .
```

LeRobot discovers the plugin through
`lerobot.utils.import_utils.register_third_party_plugins()`.

## Test

```bash
conda run -n lerobot pytest -q tests/test_latent_smolvla.py
```

The tests require `lerobot` to be installed in the active environment.

## Example Train Command

```bash
lerobot-train \
  --policy.type=latent_smolvla \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --policy.training_mode=multitask \
  --policy.latent_label_key=latent_labels \
  --policy.latent_head_mode=index_cross_entropy \
  --batch_size=8 \
  --steps=200
```

## Important Config Knobs

- `policy.training_mode` controls whether training uses action supervision, latent supervision, or both.
- `policy.latent_head_mode=index_cross_entropy` expects discrete latent labels.
- `policy.latent_head_mode=vector_diffusion` expects continuous latent vectors shaped to `latent_vector_dim`.
- `policy.latent_supervision_key` and `policy.action_supervision_key` can gate losses per sample for mixed-supervision batches.
- `policy.training_mode=latent` is latent-only training and is not intended for action inference.

## Notes

- The registered LeRobot policy key is `latent_smolvla`.
- The package import path is `lerobot_policy_latent_smolvla`.
- The policy keeps SmolVLA-style preprocessing, including tokenization and newline normalization for task strings.

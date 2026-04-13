# LeRobot Latent SmolVLA Policy

Installable third-party LeRobot policy package that registers
`policy.type=latent_smolvla`.

This package is a standalone SmolVLA variant with packed joint latent/action
diffusion. It keeps SmolVLA-style preprocessing but trains a single motion
denoiser over concatenated `[latent | action]` targets at each future step.

The installable package name is `lerobot_policy_latent_smolvla`.

## What It Adds

- one packed joint diffusion head over `[latent | action]`
- `training_mode` options: `action`, `latent`, or `multitask`
- `latent_head_mode=joint_diffusion`
- latent target routing through `latent_label_key` with default `latent_labels.continuous_vector_latents`
- latent target validity through `latent_valid_key` with default `latent_labels.valid`
- optional per-sample branch routing through `latent_supervision_key` and `action_supervision_key`
- preservation of configured latent-related and supervision keys through preprocessing via complementary data

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

For mixed-supervision batches, keep `policy.training_mode=multitask` globally and
route the branches per sample with boolean supervision masks.

```bash
lerobot-train \
  --policy.type=latent_smolvla \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --policy.training_mode=multitask \
  --policy.latent_head_mode=joint_diffusion \
  --policy.latent_label_key=latent_labels.continuous_vector_latents \
  --policy.latent_valid_key=latent_labels.valid \
  --policy.latent_supervision_key=latent_supervision \
  --policy.action_supervision_key=action_supervision \
  --batch_size=8 \
  --steps=200
```

## Important Config Knobs

- `policy.training_mode` is still a run-level switch. Use `multitask` when a batch may contain both action-supervised and latent-supervised samples.
- `policy.latent_head_mode=joint_diffusion` is the only supported mode and expects continuous latent vectors.
- Packed joint diffusion requires `latent_code_seq_len == chunk_size`.
- Each latent step must fit within `max_action_dim`; the model pads the latent half to `max_action_dim` before concatenating `[latent | action]`.
- `policy.latent_valid_key` should indicate whether the latent target is usable for a sample.
- `policy.latent_supervision_key` and `policy.action_supervision_key` are optional per-sample boolean masks that decide whether each loss branch is applied.
- the effective latent gate is `latent_valid_key AND latent_supervision_key` when both are configured.
- prefer a top-level latent namespace such as `latent_labels.*`; do not store latent labels under `observation.*` because dataset observation delta expansion will add extra temporal axes.
- `policy.training_mode=latent` is latent-only training and is not intended for action inference.

## Notes

- The registered LeRobot policy key is `latent_smolvla`.
- The package import path is `lerobot_policy_latent_smolvla`.
- The policy keeps SmolVLA-style preprocessing, including tokenization and newline normalization for task strings.

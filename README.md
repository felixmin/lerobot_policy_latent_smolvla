# LeRobot Latent SmolVLA Policy

Installable third-party LeRobot policy package that registers
`policy.type=latent_smolvla`.

This package is a standalone SmolVLA variant with hierarchical latent-plan and
action diffusion. It keeps SmolVLA-style preprocessing and the same VLM
backbone, adds a latent-plan diffusion stage, projects the predicted latent plan
into the action prefix, then runs the usual action diffusion stage conditioned
on that plan.

The installable package name is `lerobot_policy_latent_smolvla`.

## What It Adds

- a latent-plan diffusion expert before the action diffusion expert
- a separate action diffusion expert conditioned on the latent plan
- shared VLM and processor weights between the latent and action stages
- `training_mode` options: `action`, `latent`, or `multitask`
- latent target routing through `latent_label_key` with default `latent_labels.continuous_vector_latents`
- latent target validity through `latent_valid_key` with default `latent_labels.valid`
- branch routing through `latent_supervision_key` and `action_supervision_key`
- preservation of configured latent-related and supervision keys through preprocessing via complementary data

## Difference From Default SmolVLA

Default `policy.type=smolvla` directly denoises action chunks from image,
language, and state context. `policy.type=latent_smolvla` first denoises a
sequence of continuous latent vectors, turns those vectors into plan tokens, and
uses those plan tokens as extra prefix context for action denoising.

Most SmolVLA knobs are inherited unchanged, including `n_obs_steps=1`,
`chunk_size=50`, `n_action_steps=50`, `max_state_dim=32`, `max_action_dim=32`,
`resize_imgs_with_padding=(512, 512)`, `tokenizer_max_length=48`,
`num_steps=10`, `use_cache=True`, `freeze_vision_encoder=True`,
`train_expert_only=True`, `train_state_proj=True`, optimizer defaults, backbone
defaults, RTC, and `torch.compile` settings.

Inherited defaults that intentionally differ from current default SmolVLA:

| Parameter | Latent SmolVLA default | Default SmolVLA default | Why |
| --- | --- | --- | --- |
| `load_vlm_weights` | `True` | `False` | initialize the shared VLM from the pretrained backbone by default |
| `scheduler_decay_steps` | `100_000` | `30_000` | longer decay schedule for the latent/action training setup |

Parameters added on top of default SmolVLA are grouped by how often you need to
think about them. For normal mixed action/latent training, the first table is the
important one.

Core latent data and routing parameters:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `training_mode` | `"multitask"` | run-level loss mode: `"action"`, `"latent"`, or `"multitask"` |
| `latent_delta_indices` | `None` | timestamps used to load latent labels; if unset, latent sequence length is `chunk_size` |
| `max_latent_dim` | `None`, resolved to `max_action_dim` | padded width for each latent vector step |
| `latent_label_key` | `"latent_labels.continuous_vector_latents"` | batch key containing continuous latent targets |
| `latent_valid_key` | `"latent_labels.valid"` | batch key marking which latent targets are usable |
| `latent_supervision_key` | `None` | central routing mask for latent loss in mixed-supervision batches; normally set to a batch key such as `latent_supervision` |
| `action_supervision_key` | `None` | central routing mask for action loss in mixed-supervision batches; normally set to a batch key such as `action_supervision` |
| `state_conditioning` | `"action_supervised"` | when to include state tokens: `"always"`, `"never"`, or only for action-supervised samples |

Training behavior parameters:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `action_loss_weight` | `1.0` | multiplier for the action diffusion loss |
| `latent_loss_weight` | `1.0` | multiplier for the latent diffusion loss |
| `latent_teacher_force_ratio_start` | `1.0` | initial probability of feeding ground-truth latents to the action stage during training |
| `latent_teacher_force_ratio_end` | `0.0` | final teacher-forcing probability after decay |
| `latent_teacher_force_delay_steps` | `0` | number of optimizer steps before teacher-forcing decay starts |
| `latent_teacher_force_decay_steps` | `100_000` | linear decay length for teacher forcing |
| `normalize_latent_targets` | `True` | normalize latent targets in preprocessing |
| `latent_normalization_source` | `"latent"` | use latent-label stats or action stats for latent normalization |
| `latent_flow_beta_alpha` | `1.5` | alpha parameter for diffusion timestep beta sampling |
| `latent_flow_beta_beta` | `1.0` | beta parameter for diffusion timestep beta sampling |

Ablation, compatibility, and numerical-safety parameters:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `latent_head_mode` | `"vector_diffusion"` | compatibility field; this branch only supports continuous vector diffusion |
| `freeze_latent_stage` | `False` | freeze latent-stage modules while keeping the action interface trainable |
| `latent_conditioning` | `"predicted"` | ablation/debug switch; `"zeros"` conditions actions on zero plan tokens instead of predicted latent plans |
| `latent_normalization_eps` | `1e-8` | epsilon for latent target normalization |

Derived behavior: `latent_sequence_length` is `len(latent_delta_indices)` when
`latent_delta_indices` is set, otherwise it is `chunk_size`.

## Supervision Routing Contract

For the usual mixed-supervision setup, each sample should carry two boolean
routing keys: one configured by `latent_supervision_key` and one configured by
`action_supervision_key`. These keys tell the policy whether the sample trains
only the latent expert or trains both the latent and action experts.

| `latent_supervision` | `action_supervision` | Effect |
| --- | --- | --- |
| `true` | `false` | train the latent expert only |
| `true` | `true` | train both the latent expert and action expert |
| `false` | `true` | train the action expert only, if this mode is intentionally used |
| `false` | `false` | sample contributes no action or latent loss |

The effective latent mask is `latent_valid_key AND latent_supervision_key`.
`latent_valid_key` says whether the latent target is usable; `latent_supervision_key`
says whether this sample should train the latent branch. With the default
`state_conditioning="action_supervised"`, `action_supervision_key` also controls
which samples receive state tokens.

In the usual setup, these routing keys are attached by a modified dataloader or
dataset mixer based on episode sets. They can also be stored as permanent dataset
features when the split between latent-only and latent-plus-action episodes is
fixed. Leaving the keys unset makes every sample eligible for the corresponding
branch selected by `training_mode`, which is only appropriate for homogeneous
batches.

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
  --policy.latent_label_key=latent_labels.continuous_vector_latents \
  --policy.latent_valid_key=latent_labels.valid \
  --policy.latent_supervision_key=latent_supervision \
  --policy.action_supervision_key=action_supervision \
  --batch_size=8 \
  --steps=200
```

## Important Config Knobs

- `policy.training_mode` is still a run-level switch. Use `multitask` when a batch may contain both action-supervised and latent-supervised samples.
- Each latent step must fit within `max_latent_dim`; the model pads shorter latent vectors to `max_latent_dim`.
- `policy.latent_conditioning=zeros` disables predicted latent conditioning for the action stage. In action-only mode it also skips the latent expert.
- `policy.latent_valid_key` should indicate whether the latent target is usable for a sample.
- `policy.latent_supervision_key` and `policy.action_supervision_key` are the branch-routing masks for mixed-supervision batches and are expected in the usual mixed episode-set setup.
- the effective latent gate is `latent_valid_key AND latent_supervision_key` when both are configured.
- prefer a top-level latent namespace such as `latent_labels.*`; do not store latent labels under `observation.*` because dataset observation delta expansion will add extra temporal axes.
- `policy.training_mode=latent` is latent-only training and is not intended for action inference.

## Notes

- The registered LeRobot policy key is `latent_smolvla`.
- The package import path is `lerobot_policy_latent_smolvla`.
- The policy keeps SmolVLA-style preprocessing, including tokenization and newline normalization for task strings.

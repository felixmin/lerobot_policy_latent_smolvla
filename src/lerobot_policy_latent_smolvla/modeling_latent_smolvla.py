#!/usr/bin/env python

import math
from collections import deque
from typing import TypedDict

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from typing_extensions import Unpack

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.device_utils import get_safe_dtype

from lerobot_policy_latent_smolvla.configuration_latent_smolvla import LatentSmolVLAConfig
from lerobot_policy_latent_smolvla.loss_utils import (
    expand_keep_mask,
    make_sample_keep_mask,
    make_sequence_keep_mask,
    masked_mean_or_zero,
    reduce_action_per_sample,
    reshape_latent_vector_sequence,
    sample_beta_time,
)
from lerobot_policy_latent_smolvla.smolvlm_with_expert_standalone import SmolVLMWithExpertModel


class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size,)`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def make_att_2d_masks(pad_masks, att_masks):
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def resize_with_pad(img, width, height, pad_value=-1):
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))
    return F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)


def pad_vector(vector, new_dim):
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def safe_arcsin(value):
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value):
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    def linear_to_radian(linear_position, arm_length, horn_radius):
        ratio = (horn_radius**2 + linear_position**2 - arm_length**2) / (
            2 * horn_radius * linear_position
        )
        return safe_arcsin(ratio)

    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value):
    value = unnormalize(value, min_val=0.4, max_val=1.5)
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value):
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)


def pad_tensor(tensor, max_len, pad_value=0):
    batch, seq = tensor.shape[:2]
    padded_tensor = torch.full(
        (batch, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device
    )
    padded_tensor[:, :seq] = tensor
    return padded_tensor


def masked_rms(values: Tensor, keep: Tensor) -> float:
    keep = keep.bool()
    if not torch.any(keep):
        return 0.0
    selected = values.detach().to(dtype=torch.float32).masked_select(expand_keep_mask(values, keep))
    return float(torch.sqrt(torch.mean(selected.square())).cpu())


def masked_abs_mean(values: Tensor, keep: Tensor) -> float:
    keep = keep.bool()
    if not torch.any(keep):
        return 0.0
    selected = values.detach().to(dtype=torch.float32).masked_select(expand_keep_mask(values, keep))
    return float(selected.abs().mean().cpu())


def parameter_grad_l2_norm(parameters: list[nn.Parameter]) -> float:
    sq_norm = 0.0
    has_grad = False
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().to(dtype=torch.float32)
        sq_norm += float(grad.square().sum().item())
        has_grad = True
    if not has_grad:
        return 0.0
    return sq_norm**0.5


class LatentSmolVLAFlowMatching(nn.Module):
    """Two-stage SmolVLA-style core with latent-plan diffusion followed by action diffusion."""

    def __init__(self, config: LatentSmolVLAConfig, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config
        self.rtc_processor = rtc_processor

        self.latent_vlm_with_expert = SmolVLMWithExpertModel(
            model_id=self.config.vlm_model_name,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            load_vlm_weights=self.config.load_vlm_weights,
            attention_mode=self.config.attention_mode,
            num_expert_layers=self.config.num_expert_layers,
            num_vlm_layers=self.config.num_vlm_layers,
            self_attn_every_n_layers=self.config.self_attn_every_n_layers,
            expert_width_multiplier=self.config.expert_width_multiplier,
            device=self.config.device if self.config.device is not None else "auto",
        )
        self.action_vlm_with_expert = SmolVLMWithExpertModel(
            model_id=self.config.vlm_model_name,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            load_vlm_weights=False,
            attention_mode=self.config.attention_mode,
            num_expert_layers=self.config.num_expert_layers,
            num_vlm_layers=self.config.num_vlm_layers,
            self_attn_every_n_layers=self.config.self_attn_every_n_layers,
            expert_width_multiplier=self.config.expert_width_multiplier,
            device=self.config.device if self.config.device is not None else "auto",
            shared_vlm=self.latent_vlm_with_expert.vlm,
            shared_processor=self.latent_vlm_with_expert.processor,
        )
        self.state_proj = nn.Linear(
            self.config.max_state_dim, self.latent_vlm_with_expert.config.text_config.hidden_size
        )
        self.max_latent_dim = int(
            getattr(config, "max_latent_dim", config.max_action_dim)
        )
        self.prefix_hidden_size = self.latent_vlm_with_expert.config.text_config.hidden_size
        self.expert_hidden_size = self.latent_vlm_with_expert.expert_hidden_size

        self.latent_in_proj = nn.Linear(self.max_latent_dim, self.expert_hidden_size)
        self.latent_out_proj = nn.Linear(self.expert_hidden_size, self.max_latent_dim)
        self.latent_time_mlp_in = nn.Linear(self.expert_hidden_size * 2, self.expert_hidden_size)
        self.latent_time_mlp_out = nn.Linear(self.expert_hidden_size, self.expert_hidden_size)

        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.expert_hidden_size)
        self.action_out_proj = nn.Linear(self.expert_hidden_size, self.config.max_action_dim)
        self.action_time_mlp_in = nn.Linear(self.expert_hidden_size * 2, self.expert_hidden_size)
        self.action_time_mlp_out = nn.Linear(self.expert_hidden_size, self.expert_hidden_size)

        self.latent_plan_proj = nn.Linear(self.max_latent_dim, self.prefix_hidden_size)
        self.latent_anchor_mlp_in = nn.Linear(1, self.prefix_hidden_size)
        self.latent_anchor_mlp_out = nn.Linear(self.prefix_hidden_size, self.prefix_hidden_size)
        self.latent_duration_mlp_in = nn.Linear(1, self.prefix_hidden_size)
        self.latent_duration_mlp_out = nn.Linear(self.prefix_hidden_size, self.prefix_hidden_size)

        self.register_buffer("_teacher_force_step", torch.zeros((), dtype=torch.long), persistent=False)

        self.set_requires_grad()
        self.fake_image_token = self.latent_vlm_with_expert.processor.tokenizer.fake_image_token_id
        self.global_image_token = self.latent_vlm_with_expert.processor.tokenizer.global_image_token_id
        self.global_image_start_token = torch.tensor(
            [self.fake_image_token, self.global_image_token], dtype=torch.long
        )
        self.add_image_special_tokens = self.config.add_image_special_tokens
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
        self.prefix_length = self.config.prefix_length

        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj
        if getattr(self.config, "freeze_latent_stage", False):
            latent_stage_modules = [
                self.latent_vlm_with_expert.lm_expert,
                self.latent_in_proj,
                self.latent_out_proj,
                self.latent_time_mlp_in,
                self.latent_time_mlp_out,
            ]
            for module in latent_stage_modules:
                for params in module.parameters():
                    params.requires_grad = False

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, batch_size, device):
        return sample_beta_time(
            batch_size=int(batch_size),
            device=device,
            dtype=torch.float32,
            alpha=float(self.config.latent_flow_beta_alpha),
            beta=float(self.config.latent_flow_beta_beta),
        )

    def _head_prefixes(self, images, img_masks, image_keys, lang_tokens, lang_masks, state=None, state_mask=None):
        """Build the (latent, action) prefixes. Each camera is encoded once; the latent
        prefix omits any action-only camera so the latent stage never depends on it.
        latent_camera_keys / action_camera_keys default to None = all cameras."""
        tokens = {}
        for key, img, mask in zip(image_keys, images, img_masks, strict=True):
            emb = self.latent_vlm_with_expert.embed_image(img)
            tokens[key] = (emb * (emb.shape[-1] ** 0.5), mask)
        latent_keys = self.config.latent_camera_keys or image_keys
        action_keys = self.config.action_camera_keys or image_keys
        return (
            self.embed_prefix([tokens[k] for k in latent_keys], lang_tokens, lang_masks, state, state_mask),
            self.embed_prefix([tokens[k] for k in action_keys], lang_tokens, lang_masks, state, state_mask),
        )

    def embed_prefix(
        self, image_tokens, lang_tokens, lang_masks, state=None, state_mask=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assemble one head's prefix from pre-encoded image tokens + language + state.
        image_tokens is an ordered list of (image_emb, sample_mask) pairs."""
        embs, pad_masks, att_masks = [], [], []

        def special(token, batch_size):  # image start/end marker, embedded as a language token
            t = self.latent_vlm_with_expert.embed_language_tokens(
                token.to(device=self.latent_vlm_with_expert.vlm.device)
            ).unsqueeze(0).expand(batch_size, -1, -1)
            embs.append(t)
            pad_masks.append(torch.ones_like(t[:, :, 0], dtype=torch.bool))
            att_masks.extend([0] * t.shape[1])

        for img_emb, img_mask in image_tokens:
            batch_size, num_img_embs = img_emb.shape[:2]
            if self.add_image_special_tokens:
                special(self.global_image_start_token, batch_size)
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(batch_size, num_img_embs))
            att_masks.extend([0] * num_img_embs)
            if self.add_image_special_tokens:
                special(self.image_end_token, batch_size)

        lang_emb = self.latent_vlm_with_expert.embed_language_tokens(lang_tokens)
        embs.append(lang_emb * math.sqrt(lang_emb.shape[-1]))
        pad_masks.append(lang_masks)
        att_masks.extend([0] * lang_emb.shape[1])

        if state is not None:
            state_emb = self.state_proj(state)
            state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
            if state_mask is None:
                state_mask = torch.ones(state_emb.shape[:2], dtype=torch.bool, device=state_emb.device)
            else:
                state_mask = state_mask.to(device=state_emb.device, dtype=torch.bool)
                if state_mask.ndim == 1:
                    state_mask = state_mask[:, None].expand(state_emb.shape[:2])
            embs.append(state_emb)
            pad_masks.append(state_mask)
            att_masks.extend([1] * state_emb.shape[1])

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)[None, :]
        if pad_masks.shape[1] < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)
        return embs, pad_masks, att_masks.expand(pad_masks.shape[0], -1)

    def embed_latent_suffix(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent_emb = self.latent_in_proj(noisy_latents)
        device = latent_emb.device
        batch_size = latent_emb.shape[0]
        dtype = latent_emb.dtype
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        ).to(dtype=dtype)
        time_emb = time_emb[:, None, :].expand_as(latent_emb)

        latent_time_emb = torch.cat([latent_emb, time_emb], dim=2)
        latent_time_emb = self.latent_time_mlp_in(latent_time_emb)
        latent_time_emb = F.silu(latent_time_emb)
        latent_time_emb = self.latent_time_mlp_out(latent_time_emb)

        pad_masks = torch.ones(
            batch_size, latent_time_emb.shape[1], dtype=torch.bool, device=device
        )
        att_masks = torch.ones(
            batch_size, latent_time_emb.shape[1], dtype=latent_time_emb.dtype, device=device
        )
        return latent_time_emb, pad_masks, att_masks

    def embed_action_suffix(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_emb = self.action_in_proj(noisy_actions)
        device = action_emb.device
        batch_size = action_emb.shape[0]
        dtype = action_emb.dtype
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        ).to(dtype=dtype)
        time_emb = time_emb[:, None, :].expand_as(action_emb)

        action_time_emb = torch.cat([action_emb, time_emb], dim=2)
        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        pad_masks = torch.ones(
            batch_size, action_time_emb.shape[1], dtype=torch.bool, device=device
        )
        att_masks = torch.ones(
            batch_size, action_time_emb.shape[1], dtype=action_time_emb.dtype, device=device
        )
        return action_time_emb, pad_masks, att_masks

    def prepare_diffusion_noise(
        self,
        noise: torch.Tensor | None,
        *,
        batch_size: int,
        sequence_length: int,
        feature_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        shape = (batch_size, sequence_length, feature_dim)
        if noise is None:
            return self.sample_noise(shape, device)
        noise = noise.to(device=device, dtype=dtype)
        if tuple(noise.shape) != shape:
            raise ValueError(f"Expected diffusion noise shaped {shape}, got {tuple(noise.shape)}")
        return noise

    def split_noise(self, noise: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor] | None):
        if isinstance(noise, (tuple, list)):
            if len(noise) != 2:
                raise ValueError("Expected noise tuple/list of length 2: (latent_noise, action_noise)")
            return noise[0], noise[1]
        return None, noise

    def split_time(self, time: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor] | None):
        if isinstance(time, (tuple, list)):
            if len(time) != 2:
                raise ValueError("Expected time tuple/list of length 2: (latent_time, action_time)")
            return time[0], time[1]
        return time, time

    def make_latent_plan_mask(
        self,
        latent_valid: torch.Tensor | None,
        *,
        batch_size: int,
        sequence_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        if latent_valid is None:
            return torch.ones(
                batch_size,
                sequence_length,
                dtype=torch.bool,
                device=device,
            )
        latent_valid = latent_valid.to(device=device, dtype=torch.bool)
        if latent_valid.ndim == 1:
            return latent_valid[:, None].expand(batch_size, sequence_length)
        if latent_valid.ndim == 2 and int(latent_valid.shape[1]) == int(sequence_length):
            return latent_valid
        raise ValueError(
            "Expected latent_valid shaped [B] or [B, latent_sequence_length], "
            f"got {tuple(latent_valid.shape)}"
        )

    def embed_latent_plan(
        self,
        latent_plan: torch.Tensor,
        *,
        latent_valid: torch.Tensor | None = None,
        action_horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, _ = latent_plan.shape
        device = latent_plan.device
        dtype = latent_plan.dtype

        content_emb = self.latent_plan_proj(latent_plan)
        anchor_steps = torch.div(
            torch.arange(sequence_length, device=device, dtype=torch.long) * action_horizon,
            sequence_length,
            rounding_mode="floor",
        )
        next_anchor_steps = torch.div(
            torch.arange(1, sequence_length + 1, device=device, dtype=torch.long) * action_horizon,
            sequence_length,
            rounding_mode="floor",
        )
        duration_steps = next_anchor_steps - anchor_steps

        anchor_norm = anchor_steps.to(dtype=dtype)[None, :, None] / float(max(action_horizon - 1, 1))
        duration_norm = duration_steps.to(dtype=dtype)[None, :, None] / float(max(action_horizon, 1))

        anchor_emb = self.latent_anchor_mlp_in(anchor_norm)
        anchor_emb = F.silu(anchor_emb)
        anchor_emb = self.latent_anchor_mlp_out(anchor_emb)

        duration_emb = self.latent_duration_mlp_in(duration_norm)
        duration_emb = F.silu(duration_emb)
        duration_emb = self.latent_duration_mlp_out(duration_emb)

        plan_embs = content_emb + anchor_emb + duration_emb
        plan_pad_masks = self.make_latent_plan_mask(
            latent_valid,
            batch_size=batch_size,
            sequence_length=sequence_length,
            device=device,
        )
        plan_att_masks = torch.zeros(batch_size, sequence_length, dtype=torch.bool, device=device)
        return plan_embs, plan_pad_masks, plan_att_masks

    def teacher_force_ratio(self) -> float:
        if not self.training:
            return float(self.config.latent_teacher_force_ratio_end)
        decay_steps = int(self.config.latent_teacher_force_decay_steps)
        delay_steps = int(getattr(self.config, "latent_teacher_force_delay_steps", 0))
        step = float(self._teacher_force_step.item())
        start = float(self.config.latent_teacher_force_ratio_start)
        end = float(self.config.latent_teacher_force_ratio_end)
        if step < float(delay_steps):
            return start
        if decay_steps <= 0:
            return end
        progress = min((step - float(delay_steps)) / float(decay_steps), 1.0)
        return start + (end - start) * progress

    def forward(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        image_keys: list | None,
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor | None,
        actions: torch.Tensor | None,
        latent_vectors: torch.Tensor | None,
        latent_valid: torch.Tensor | None = None,
        state_mask: torch.Tensor | None = None,
        noise: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
        time: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if lang_tokens is not None:
            batch_size = int(lang_tokens.shape[0])
            device = lang_tokens.device
        elif state is not None:
            batch_size = int(state.shape[0])
            device = state.device
        elif actions is not None:
            batch_size = int(actions.shape[0])
            device = actions.device
        elif latent_vectors is not None:
            batch_size = int(latent_vectors.shape[0])
            device = latent_vectors.device
        else:
            raise ValueError("Cannot infer batch size without language, state, action, or latent tensors")
        dtype = torch.float32

        if actions is None:
            action_targets = torch.zeros(
                batch_size,
                self.config.chunk_size,
                self.config.max_action_dim,
                device=device,
                dtype=dtype,
            )
        else:
            action_targets = actions.to(device=device, dtype=dtype)

        if latent_vectors is None:
            latent_sequence_length = int(self.config.latent_sequence_length)
            latent_targets = torch.zeros(
                batch_size,
                latent_sequence_length,
                self.max_latent_dim,
                device=device,
                dtype=dtype,
            )
        else:
            latent_targets = reshape_latent_vector_sequence(
                latent_vectors,
                latent_sequence_length=int(self.config.latent_sequence_length)
                if self.config.latent_delta_indices is not None
                else None,
            ).to(device=device, dtype=dtype)
            latent_targets = torch.nan_to_num(latent_targets, nan=0.0, posinf=0.0, neginf=0.0)
            if int(latent_targets.shape[2]) > int(self.max_latent_dim):
                raise ValueError(
                    "Hierarchical latent diffusion requires latent step dim <= max_latent_dim, "
                    f"got latent_step_dim={latent_targets.shape[2]} max_latent_dim={self.max_latent_dim}"
                )
        if int(latent_targets.shape[-1]) != int(self.max_latent_dim):
            latent_targets = pad_vector(latent_targets, self.max_latent_dim)

        latent_noise_input, action_noise_input = self.split_noise(noise)
        latent_time, action_time = self.split_time(time)

        latent_noise = self.prepare_diffusion_noise(
            latent_noise_input,
            batch_size=batch_size,
            sequence_length=int(latent_targets.shape[1]),
            feature_dim=self.max_latent_dim,
            device=device,
            dtype=dtype,
        )
        action_noise = self.prepare_diffusion_noise(
            action_noise_input,
            batch_size=batch_size,
            sequence_length=self.config.chunk_size,
            feature_dim=self.config.max_action_dim,
            device=device,
            dtype=dtype,
        )
        if latent_time is None:
            latent_time = self.sample_time(batch_size, device)
        if action_time is None:
            action_time = self.sample_time(batch_size, device)

        action_time_expanded = action_time[:, None, None]
        action_x_t = action_time_expanded * action_noise + (1 - action_time_expanded) * action_targets
        action_u_t = action_noise - action_targets

        (
            (latent_obs_embs, latent_obs_pad_masks, latent_obs_att_masks),
            (action_obs_embs, action_obs_pad_masks, action_obs_att_masks),
        ) = self._head_prefixes(
            images, img_masks, image_keys, lang_tokens, lang_masks, state=state, state_mask=state_mask
        )

        latent_conditioning = getattr(self.config, "latent_conditioning", "predicted")
        needs_latent_loss = (
            getattr(self.config, "training_mode", "multitask") in {"latent", "multitask"}
            and float(getattr(self.config, "latent_loss_weight", 1.0)) > 0.0
        )
        run_latent_stage = latent_conditioning != "zeros" or needs_latent_loss

        if run_latent_stage:
            latent_time_expanded = latent_time[:, None, None]
            latent_x_t = latent_time_expanded * latent_noise + (1 - latent_time_expanded) * latent_targets
            latent_u_t = latent_noise - latent_targets

            latent_suffix_embs, latent_suffix_pad_masks, latent_suffix_att_masks = self.embed_latent_suffix(
                latent_x_t, latent_time
            )
            latent_pad_masks = torch.cat([latent_obs_pad_masks, latent_suffix_pad_masks], dim=1)
            latent_att_masks = torch.cat([latent_obs_att_masks, latent_suffix_att_masks], dim=1)
            latent_att_2d_masks = make_att_2d_masks(latent_pad_masks, latent_att_masks)
            latent_position_ids = torch.cumsum(latent_pad_masks, dim=1) - 1
            (_, latent_suffix_out), _ = self.latent_vlm_with_expert.forward(
                attention_mask=latent_att_2d_masks,
                position_ids=latent_position_ids,
                past_key_values=None,
                inputs_embeds=[latent_obs_embs, latent_suffix_embs],
                use_cache=False,
                fill_kv_cache=False,
            )
            latent_suffix_out = latent_suffix_out[:, -int(latent_targets.shape[1]) :].to(dtype=torch.float32)
            latent_v_t = self.latent_out_proj(latent_suffix_out)
            latent_losses = F.mse_loss(latent_u_t, latent_v_t, reduction="none")
            latent_hat = latent_x_t - latent_time_expanded * latent_v_t
        else:
            latent_losses = torch.zeros_like(latent_targets)
            latent_hat = torch.zeros_like(latent_targets)

        if latent_conditioning == "zeros":
            latent_condition = torch.zeros_like(latent_targets)
        elif self.training:
            teacher_force_ratio = self.teacher_force_ratio()
            teacher_force_mask = (
                torch.rand(batch_size, device=device) < teacher_force_ratio
            )[:, None, None]
            latent_condition = torch.where(teacher_force_mask, latent_targets, latent_hat)
            self._teacher_force_step.add_(1)
        else:
            latent_condition = latent_hat

        plan_embs, plan_pad_masks, plan_att_masks = self.embed_latent_plan(
            latent_condition,
            latent_valid=latent_valid,
            action_horizon=self.config.chunk_size,
        )
        action_prefix_embs = torch.cat([action_obs_embs, plan_embs], dim=1)
        action_prefix_pad_masks = torch.cat([action_obs_pad_masks, plan_pad_masks], dim=1)
        action_prefix_att_masks = torch.cat([action_obs_att_masks, plan_att_masks], dim=1)

        action_suffix_embs, action_suffix_pad_masks, action_suffix_att_masks = self.embed_action_suffix(
            action_x_t, action_time
        )
        action_pad_masks = torch.cat([action_prefix_pad_masks, action_suffix_pad_masks], dim=1)
        action_att_masks = torch.cat([action_prefix_att_masks, action_suffix_att_masks], dim=1)
        action_att_2d_masks = make_att_2d_masks(action_pad_masks, action_att_masks)
        action_position_ids = torch.cumsum(action_pad_masks, dim=1) - 1
        (_, action_suffix_out), _ = self.action_vlm_with_expert.forward(
            attention_mask=action_att_2d_masks,
            position_ids=action_position_ids,
            past_key_values=None,
            inputs_embeds=[action_prefix_embs, action_suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        action_suffix_out = action_suffix_out[:, -self.config.chunk_size :].to(dtype=torch.float32)
        action_v_t = self.action_out_proj(action_suffix_out)
        action_losses = F.mse_loss(action_u_t, action_v_t, reduction="none")
        return action_losses, latent_losses

    def sample_actions(
        self,
        images,
        img_masks,
        image_keys,
        lang_tokens,
        lang_masks,
        state,
        state_mask=None,
        noise=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        batch_size = lang_tokens.shape[0]
        device = state.device if state is not None else lang_tokens.device
        latent_noise_input, action_noise_input = self.split_noise(noise)
        action_noise = self.prepare_diffusion_noise(
            action_noise_input,
            batch_size=batch_size,
            sequence_length=self.config.chunk_size,
            feature_dim=self.config.max_action_dim,
            device=device,
            dtype=torch.float32,
        )

        (
            (latent_obs_embs, latent_obs_pad_masks, latent_obs_att_masks),
            (action_obs_embs, action_obs_pad_masks, action_obs_att_masks),
        ) = self._head_prefixes(
            images, img_masks, image_keys, lang_tokens, lang_masks, state=state, state_mask=state_mask
        )

        num_steps = self.config.num_steps
        dt = -1.0 / num_steps

        if getattr(self.config, "latent_conditioning", "predicted") == "zeros":
            latent_x_t = torch.zeros(
                batch_size,
                int(self.config.latent_sequence_length),
                self.max_latent_dim,
                device=device,
                dtype=torch.float32,
            )
        else:
            latent_noise = self.prepare_diffusion_noise(
                latent_noise_input,
                batch_size=batch_size,
                sequence_length=int(self.config.latent_sequence_length),
                feature_dim=self.max_latent_dim,
                device=device,
                dtype=torch.float32,
            )
            latent_prefix_att_2d_masks = make_att_2d_masks(latent_obs_pad_masks, latent_obs_att_masks)
            latent_prefix_position_ids = torch.cumsum(latent_obs_pad_masks, dim=1) - 1
            _, latent_past_key_values = self.latent_vlm_with_expert.forward(
                attention_mask=latent_prefix_att_2d_masks,
                position_ids=latent_prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[latent_obs_embs, None],
                use_cache=self.config.use_cache,
                fill_kv_cache=True,
            )

            latent_x_t = latent_noise
            for step in range(num_steps):
                current_time = 1.0 + step * dt
                latent_time = torch.tensor(current_time, dtype=torch.float32, device=device).expand(batch_size)
                latent_v_t = self.denoise_latent_step(
                    prefix_pad_masks=latent_obs_pad_masks,
                    past_key_values=latent_past_key_values,
                    x_t=latent_x_t,
                    timestep=latent_time,
                )
                latent_x_t = latent_x_t + dt * latent_v_t

        plan_embs, plan_pad_masks, plan_att_masks = self.embed_latent_plan(
            latent_x_t,
            latent_valid=None,
            action_horizon=self.config.chunk_size,
        )
        action_prefix_embs = torch.cat([action_obs_embs, plan_embs], dim=1)
        action_prefix_pad_masks = torch.cat([action_obs_pad_masks, plan_pad_masks], dim=1)
        action_prefix_att_masks = torch.cat([action_obs_att_masks, plan_att_masks], dim=1)
        action_prefix_att_2d_masks = make_att_2d_masks(action_prefix_pad_masks, action_prefix_att_masks)
        action_prefix_position_ids = torch.cumsum(action_prefix_pad_masks, dim=1) - 1
        _, action_past_key_values = self.action_vlm_with_expert.forward(
            attention_mask=action_prefix_att_2d_masks,
            position_ids=action_prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[action_prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )

        action_x_t = action_noise
        for step in range(num_steps):
            current_time = 1.0 + step * dt
            action_time = torch.tensor(current_time, dtype=torch.float32, device=device).expand(batch_size)

            def denoise_step_partial_call(input_x_t, current_timestep=action_time):
                return self.denoise_action_step(
                    prefix_pad_masks=action_prefix_pad_masks,
                    past_key_values=action_past_key_values,
                    x_t=input_x_t,
                    timestep=current_timestep,
                )

            if self._rtc_enabled():
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")
                v_t = self.rtc_processor.denoise_step(
                    x_t=action_x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=current_time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(action_x_t)

            action_x_t = action_x_t + dt * v_t

            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=current_time, x_t=action_x_t, v_t=v_t)

        return action_x_t

    def denoise_latent_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_latent_suffix(x_t, timestep)
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.latent_vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1][:, -int(x_t.shape[1]) :].to(dtype=torch.float32)
        return self.latent_out_proj(suffix_out)

    def denoise_action_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_action_suffix(x_t, timestep)
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.action_vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1][:, -self.config.chunk_size :].to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)


class LatentSmolVLAPolicy(PreTrainedPolicy):
    """Standalone SmolVLA-derived policy with optional latent supervision."""

    config_class = LatentSmolVLAConfig
    name = "latent_smolvla"

    def __init__(self, config: LatentSmolVLAConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = LatentSmolVLAFlowMatching(config, rtc_processor=self.rtc_processor)
        self.reset()

    def reset(self):
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def init_rtc_processor(self):
        self.rtc_processor = None
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)
            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def get_optim_params(self) -> dict:
        return self.parameters()

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _check_get_actions_condition(self) -> bool:
        return len(self._queues[ACTION]) == 0

    def _routed_camera_keys(self):
        """Cameras to load: the union of the two heads' lists (each None = all cameras)."""
        all_keys = list(self.config.image_features.keys())
        latent = self.config.latent_camera_keys or all_keys
        action = self.config.action_camera_keys or all_keys
        for name, keys in (("latent_camera_keys", latent), ("action_camera_keys", action)):
            unknown = [k for k in keys if k not in all_keys]
            if unknown:
                raise ValueError(f"{name} references cameras absent from image_features: {unknown}. Available: {all_keys}")
        return list(dict.fromkeys([*latent, *action]))

    def prepare_images(self, batch):
        # Load the routed cameras in a fixed order; a camera absent from a sample (e.g.
        # the extra cam on a no-extra-cam episode) is zero-filled and masked out so
        # mixed batches collate. The data pipeline should still provide a padding mask.
        keys = self._routed_camera_keys()
        images, img_masks, reference = [], [], None
        for key in keys:
            if key in batch:
                img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
                if self.config.resize_imgs_with_padding is not None:
                    img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)
                img = img * 2.0 - 1.0
                # Padding mask lives under the camera short name (e.g. observation.images.wrist
                # -> wrist_padding_mask) so it is not classified/normalized as a STATE feature.
                pm = batch.get(f"{key.rsplit('.', 1)[-1]}_padding_mask", batch.get(f"{key}_padding_mask"))
                mask = pm.reshape(img.shape[0]).bool() if pm is not None \
                    else torch.ones(img.shape[0], dtype=torch.bool, device=img.device)
                reference = img
            else:
                img, mask = None, None
            images.append(img)
            img_masks.append(mask)

        if reference is None:
            raise ValueError(f"None of the routed cameras {keys} are in the batch (keys: {list(batch)})")
        for i, img in enumerate(images):
            if img is None:
                images[i] = torch.full_like(reference, -1.0)
                img_masks[i] = torch.zeros(reference.shape[0], dtype=torch.bool, device=reference.device)
        return images, img_masks, keys

    def prepare_state(self, batch):
        if getattr(self.config, "state_conditioning", "always") == "never":
            return None
        state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        return pad_vector(state, self.config.max_state_dim)

    def prepare_state_mask(self, batch, *, batch_size: int, device: torch.device) -> Tensor | None:
        state_conditioning = getattr(self.config, "state_conditioning", "always")
        if state_conditioning == "never":
            return None
        if state_conditioning == "always":
            return torch.ones(batch_size, dtype=torch.bool, device=device)
        key = self.config.action_supervision_key
        if key is None or key not in batch:
            return torch.ones(batch_size, dtype=torch.bool, device=device)
        return make_sample_keep_mask(batch, key=key, batch_size=batch_size, device=device)

    def prepare_action(self, batch):
        return pad_vector(batch[ACTION], self.config.max_action_dim)

    def _make_action_keep_mask(
        self,
        batch: dict[str, Tensor],
        *,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        return make_sample_keep_mask(
            batch,
            key=self.config.action_supervision_key,
            batch_size=batch_size,
            device=device,
        )

    def _make_latent_keep_mask(
        self,
        batch: dict[str, Tensor],
        *,
        batch_size: int,
        sequence_length: int,
        device: torch.device,
        valid_samples: Tensor | None = None,
    ) -> Tensor:
        keep = make_sequence_keep_mask(
            batch,
            key=self.config.latent_valid_key,
            batch_size=batch_size,
            sequence_length=sequence_length,
            device=device,
        )
        keep = keep & make_sequence_keep_mask(
            batch,
            key=self.config.latent_supervision_key,
            batch_size=batch_size,
            sequence_length=sequence_length,
            device=device,
        )
        if valid_samples is not None:
            valid_samples = valid_samples.bool()
            if valid_samples.ndim == 1:
                keep = keep & valid_samples.reshape(batch_size, 1).expand(batch_size, sequence_length)
            elif valid_samples.ndim == 2 and int(valid_samples.shape[1]) == int(sequence_length):
                keep = keep & valid_samples
            else:
                raise ValueError(
                    "valid_samples must be shaped [B] or [B,S] for latent sequence masking, "
                    f"got {tuple(valid_samples.shape)}"
                )
        return keep

    def _compute_joint_vector_target_metrics(
        self,
        batch: dict[str, Tensor],
        labels: Tensor,
        *,
        action_keep: Tensor,
        latent_keep: Tensor,
    ) -> dict[str, float]:
        batch_size = int(labels.shape[0])
        latent_targets = reshape_latent_vector_sequence(
            labels,
            latent_sequence_length=int(latent_keep.shape[1]),
        ).to(dtype=torch.float32)
        sequence_length = int(latent_targets.shape[1])
        latent_sample_keep = latent_keep.bool().any(dim=1)
        joint_sample_keep = action_keep.bool() & latent_sample_keep
        joint_count = int(joint_sample_keep.sum().item())
        metrics: dict[str, float] = {
            "batch_joint_supervised_samples": float(joint_count),
            "batch_joint_supervised_denominator": float(batch_size),
            "batch_joint_supervised_fraction": float(joint_count) / float(batch_size)
            if batch_size > 0
            else 0.0,
            "batch_joint_supervised_tokens": 0.0,
            "batch_joint_supervised_token_denominator": float(batch_size * sequence_length),
            "batch_joint_supervised_token_fraction": 0.0,
        }

        if ACTION not in batch:
            return metrics

        raw_actions = batch[ACTION].to(device=labels.device, dtype=torch.float32)
        action_sequence_length = int(raw_actions.shape[1])
        action_step_keep = action_keep.bool().reshape(batch_size, 1).expand(
            batch_size, action_sequence_length
        )
        if "action_is_pad" in batch:
            action_step_keep = action_step_keep & (
                ~batch["action_is_pad"].to(device=labels.device, dtype=torch.bool)
            )
        metrics["action_target_rms"] = masked_rms(raw_actions, action_step_keep)
        metrics["action_target_abs_mean"] = masked_abs_mean(raw_actions, action_step_keep)
        metrics["latent_target_rms"] = masked_rms(latent_targets, latent_keep)
        metrics["latent_target_abs_mean"] = masked_abs_mean(latent_targets, latent_keep)

        metrics["joint_action_latent_target_seq_len_match"] = float(
            action_sequence_length == sequence_length
        )
        if action_sequence_length != sequence_length:
            return metrics

        joint_step_keep = action_step_keep & latent_keep.bool()
        joint_token_count = int(joint_step_keep.sum().item())
        metrics["batch_joint_supervised_tokens"] = float(joint_token_count)
        metrics["batch_joint_supervised_token_fraction"] = (
            float(joint_token_count) / float(batch_size * sequence_length)
            if batch_size > 0 and sequence_length > 0
            else 0.0
        )
        if joint_token_count == 0:
            return metrics

        padded_actions = self.prepare_action(batch).to(device=labels.device, dtype=torch.float32)

        metrics["joint_action_latent_target_numel_match_raw"] = float(
            raw_actions.shape[-1] == latent_targets.shape[-1]
        )
        metrics["joint_action_latent_target_numel_match_padded"] = float(
            padded_actions.shape[-1] == latent_targets.shape[-1]
        )

        action_targets = None
        if raw_actions.shape[-1] == latent_targets.shape[-1]:
            action_targets = raw_actions
        elif padded_actions.shape[-1] == latent_targets.shape[-1]:
            action_targets = padded_actions

        if action_targets is None:
            return metrics

        joint_action = action_targets[joint_step_keep]
        joint_latent = latent_targets[joint_step_keep]
        diff = joint_action - joint_latent
        cosine = F.cosine_similarity(joint_action, joint_latent, dim=1, eps=1e-8)
        metrics["joint_action_latent_target_mse"] = float(diff.square().mean().detach().cpu())
        metrics["joint_action_latent_target_abs_mean"] = float(diff.abs().mean().detach().cpu())
        metrics["joint_action_latent_target_cosine"] = float(cosine.mean().detach().cpu())
        return metrics

    def get_gradient_metrics(self) -> dict[str, float]:
        planner_prefixes = (
            "model.latent_in_proj",
            "model.latent_out_proj",
            "model.latent_time_mlp_in",
            "model.latent_time_mlp_out",
            "model.latent_plan_proj",
            "model.latent_anchor_mlp_in",
            "model.latent_anchor_mlp_out",
            "model.latent_duration_mlp_in",
            "model.latent_duration_mlp_out",
        )
        action_prefixes = (
            "model.action_in_proj",
            "model.action_out_proj",
            "model.action_time_mlp_in",
            "model.action_time_mlp_out",
        )

        planner_parameters: list[nn.Parameter] = []
        action_parameters: list[nn.Parameter] = []
        backbone_parameters: list[nn.Parameter] = []
        all_parameters: list[nn.Parameter] = []

        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            all_parameters.append(parameter)
            if name.startswith(planner_prefixes):
                planner_parameters.append(parameter)
            elif name.startswith(action_prefixes):
                action_parameters.append(parameter)
            else:
                backbone_parameters.append(parameter)

        planner_norm = parameter_grad_l2_norm(planner_parameters)
        action_norm = parameter_grad_l2_norm(action_parameters)
        backbone_norm = parameter_grad_l2_norm(backbone_parameters)

        return {
            "grad_norm_model_total": parameter_grad_l2_norm(all_parameters),
            "grad_norm_shared": backbone_norm,
            "grad_norm_planner_head": planner_norm,
            "grad_norm_action_head": action_norm,
            "grad_norm_backbone": backbone_norm,
            "grad_norm_latent_motion": planner_norm,
            "grad_norm_action_motion": action_norm,
        }

    def _pi_aloha_decode_state(self, state):
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        return state

    def _pi_aloha_encode_actions(self, actions):
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions

    def forward(
        self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"
    ) -> tuple[Tensor, dict]:
        if reduction != "mean":
            raise NotImplementedError("latent_smolvla currently supports reduction='mean' only")

        if self.config.adapt_to_pi_aloha and OBS_STATE in batch and ACTION in batch:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        latent_key = str(self.config.latent_label_key)
        if self.config.training_mode in {"action", "multitask"} and ACTION not in batch:
            raise KeyError(f"Missing action batch key {ACTION!r}")
        if self.config.training_mode in {"latent", "multitask"} and latent_key not in batch:
            raise KeyError(f"Missing latent label batch key {latent_key!r}")

        images, img_masks, image_keys = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        batch_size = int(lang_tokens.shape[0])
        state_mask = self.prepare_state_mask(
            batch,
            batch_size=batch_size,
            device=lang_tokens.device,
        )
        batch_device = state.device if state is not None else lang_tokens.device
        latent_vector_labels = (
            batch[latent_key].to(device=batch_device, dtype=torch.float32)
            if latent_key in batch
            else None
        )
        latent_sequence_length = int(
            getattr(self.config, "latent_sequence_length", getattr(self.config, "chunk_size", 1))
        )
        if latent_vector_labels is not None and latent_vector_labels.ndim >= 3:
            latent_sequence_length = int(latent_vector_labels.shape[1])
        elif self.config.latent_valid_key and self.config.latent_valid_key in batch:
            latent_valid_values = batch[self.config.latent_valid_key]
            if torch.is_tensor(latent_valid_values) and latent_valid_values.ndim == 2:
                latent_sequence_length = int(latent_valid_values.shape[1])

        latent_valid = None
        if self.config.latent_valid_key and self.config.latent_valid_key in batch:
            latent_valid = make_sequence_keep_mask(
                batch,
                key=self.config.latent_valid_key,
                batch_size=batch_size,
                sequence_length=latent_sequence_length,
                device=lang_tokens.device,
            )
        latent_is_pad_key = f"{latent_key}_is_pad"
        if latent_is_pad_key in batch:
            latent_not_pad_values = batch[latent_is_pad_key]
            if torch.is_tensor(latent_not_pad_values):
                latent_not_pad_values = ~latent_not_pad_values.to(
                    device=lang_tokens.device,
                    dtype=torch.bool,
                )
            else:
                latent_not_pad_values = ~torch.as_tensor(
                    latent_not_pad_values,
                    device=lang_tokens.device,
                    dtype=torch.bool,
                )
            latent_not_pad = make_sequence_keep_mask(
                {latent_is_pad_key: latent_not_pad_values},
                key=latent_is_pad_key,
                batch_size=batch_size,
                sequence_length=latent_sequence_length,
                device=lang_tokens.device,
            )
            latent_valid = latent_not_pad if latent_valid is None else latent_valid & latent_not_pad
        actions = self.prepare_action(batch) if ACTION in batch else None
        action_is_pad = batch.get("action_is_pad")
        action_losses, latent_losses = self.model(
            images,
            img_masks,
            image_keys,
            lang_tokens,
            lang_masks,
            state,
            actions,
            latent_vector_labels,
            latent_valid=latent_valid,
            state_mask=state_mask,
            noise=noise,
            time=time,
        )
        batch_size = int(action_losses.shape[0])
        action_keep = (
            self._make_action_keep_mask(
                batch,
                batch_size=batch_size,
                device=action_losses.device,
            )
            if self.config.training_mode in {"action", "multitask"}
            else torch.zeros(batch_size, dtype=torch.bool, device=action_losses.device)
        )
        latent_keep = (
            self._make_latent_keep_mask(
                batch,
                batch_size=batch_size,
                sequence_length=int(latent_losses.shape[1]),
                device=latent_losses.device,
                valid_samples=latent_valid,
            )
            if self.config.training_mode in {"latent", "multitask"}
            else torch.zeros(
                batch_size,
                int(latent_losses.shape[1]),
                dtype=torch.bool,
                device=latent_losses.device,
            )
        )

        per_sample_action = reduce_action_per_sample(
            action_losses,
            feature_dim=self.config.max_action_dim,
            action_is_pad=action_is_pad,
        )
        latent_is_pad = ~latent_keep
        if action_is_pad is not None and int(action_is_pad.shape[1]) == int(latent_is_pad.shape[1]):
            latent_is_pad = latent_is_pad | action_is_pad.to(device=latent_is_pad.device, dtype=torch.bool)
        per_sample_latent = reduce_action_per_sample(
            latent_losses,
            feature_dim=getattr(
                self.model,
                "max_latent_dim",
                getattr(self.config, "max_latent_dim", self.config.max_action_dim),
            ),
            action_is_pad=latent_is_pad,
        )
        action_loss, action_kept = masked_mean_or_zero(per_sample_action, action_keep)
        latent_sample_keep = latent_keep.any(dim=1)
        latent_loss, latent_kept_samples = masked_mean_or_zero(per_sample_latent, latent_sample_keep)
        latent_kept_tokens = int(latent_keep.sum().item())
        latent_token_denominator = int(latent_losses.shape[0]) * int(latent_losses.shape[1])

        metrics: dict[str, float | Tensor] = {
            "action_loss": float(action_loss.detach().cpu()),
            "action_supervised_samples": float(action_kept),
            "batch_action_supervised_samples": float(action_kept),
            "batch_action_supervised_denominator": float(int(per_sample_action.shape[0])),
            "batch_action_supervised_fraction": (
                float(action_kept) / float(int(per_sample_action.shape[0]))
                if int(per_sample_action.shape[0]) > 0
                else 0.0
            ),
            "_action_supervised_denominator": float(int(per_sample_action.shape[0])),
            "_action_loss_denominator_exact": float(action_kept),
            "_action_loss_tensor": action_loss,
            "latent_loss": float(latent_loss.detach().cpu()),
            "latent_supervised_samples": float(latent_kept_samples),
            "batch_latent_supervised_samples": float(latent_kept_samples),
            "batch_latent_supervised_denominator": float(int(per_sample_latent.shape[0])),
            "batch_latent_supervised_fraction": (
                float(latent_kept_samples) / float(int(per_sample_latent.shape[0]))
                if int(per_sample_latent.shape[0]) > 0
                else 0.0
            ),
            "_latent_supervised_denominator": float(int(per_sample_latent.shape[0])),
            "_latent_loss_denominator_exact": float(latent_kept_samples),
            "_latent_loss_tensor": latent_loss,
            "batch_latent_supervised_tokens": float(latent_kept_tokens),
            "batch_latent_supervised_token_denominator": float(latent_token_denominator),
            "batch_latent_supervised_token_fraction": (
                float(latent_kept_tokens) / float(latent_token_denominator)
                if latent_token_denominator > 0
                else 0.0
            ),
        }
        weighted_action = (
            float(self.config.action_loss_weight) * action_loss
            if self.config.training_mode in {"action", "multitask"}
            else action_loss * 0.0
        )
        weighted_latent = (
            float(self.config.latent_loss_weight) * latent_loss
            if self.config.training_mode in {"latent", "multitask"}
            else latent_loss * 0.0
        )
        total = weighted_action + weighted_latent
        weighted_action_value = float(weighted_action.detach().cpu())
        weighted_latent_value = float(weighted_latent.detach().cpu())

        if (
            latent_vector_labels is not None
            and action_keep is not None
            and latent_keep is not None
        ):
            metrics.update(
                self._compute_joint_vector_target_metrics(
                    batch,
                    latent_vector_labels,
                    action_keep=action_keep,
                    latent_keep=latent_keep,
                )
            )

        total_value = float(total.detach().cpu())
        metrics["loss"] = total_value
        metrics["action_loss_weighted"] = weighted_action_value
        metrics["latent_loss_weighted"] = weighted_latent_value
        metrics["loss_weighted_action_fraction"] = (
            weighted_action_value / total_value if total_value > 0.0 else 0.0
        )
        metrics["loss_weighted_latent_fraction"] = (
            weighted_latent_value / total_value if total_value > 0.0 else 0.0
        )
        metrics["mode_action"] = float(self.config.training_mode == "action")
        metrics["mode_latent"] = float(self.config.training_mode == "latent")
        metrics["mode_multitask"] = float(self.config.training_mode == "multitask")
        return total, metrics

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.config.adapt_to_pi_aloha and OBS_STATE in batch:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
        return batch

    def _get_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        for key in batch:
            if key in self._queues and key != ACTION:
                batch[key] = torch.stack(list(self._queues[key]), dim=1)

        images, img_masks, image_keys = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        state_mask = self.prepare_state_mask(
            batch,
            batch_size=int(lang_tokens.shape[0]),
            device=lang_tokens.device,
        )
        actions = self.model.sample_actions(
            images, img_masks, image_keys, lang_tokens, lang_masks, state, state_mask=state_mask, noise=noise, **kwargs
        )

        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]
        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)
        return actions

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        if self.config.training_mode == "latent":
            raise NotImplementedError("training_mode='latent' is for latent supervision only, not action inference")
        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        return self._get_action_chunk(batch, noise, **kwargs)

    @torch.no_grad()
    def select_action(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        if self.config.training_mode == "latent":
            raise NotImplementedError("training_mode='latent' is for latent supervision only, not action inference")
        assert not self._rtc_enabled(), "RTC is not supported for select_action, use it with predict_action_chunk"

        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        if self._check_get_actions_condition():
            actions = self._get_action_chunk(batch, noise, **kwargs)
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])
        return self._queues[ACTION].popleft()

    def _get_default_peft_targets(self) -> dict[str, any]:
        freeze_latent_stage = getattr(getattr(self, "config", None), "freeze_latent_stage", False)
        common_projections = ["state_proj"]
        if not freeze_latent_stage:
            common_projections.extend(
                [
                    "latent_in_proj",
                    "latent_out_proj",
                    "latent_time_mlp_in",
                    "latent_time_mlp_out",
                ]
            )
        common_projections.extend(
            [
                "latent_plan_proj",
                "latent_anchor_mlp_in",
                "latent_anchor_mlp_out",
                "latent_duration_mlp_in",
                "latent_duration_mlp_out",
                "action_in_proj",
                "action_out_proj",
                "action_time_mlp_in",
                "action_time_mlp_out",
            ]
        )
        common_projection_pattern = "|".join(common_projections)
        expert_pattern = "action_vlm_with_expert"
        if not freeze_latent_stage:
            expert_pattern = "latent_vlm_with_expert|action_vlm_with_expert"
        target_modules = (
            rf"(model\.({expert_pattern})\.lm_expert\..*\.(q|v)_proj|model\.({common_projection_pattern}))"
        )
        return {
            "target_modules": target_modules,
            "modules_to_save": [],
        }

    def _validate_peft_config(self, peft_config) -> None:
        super()._validate_peft_config(peft_config)
        if not self.config.load_vlm_weights:
            import logging

            logging.warning(
                "Training latent_smolvla from scratch using PEFT. This is unlikely to yield good results. "
                "Set `load_vlm_weights=True` to fine-tune the existing policy."
            )

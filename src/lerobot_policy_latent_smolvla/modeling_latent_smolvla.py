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
    """Self-contained SmolVLA-style core with packed joint latent/action diffusion."""

    def __init__(self, config: LatentSmolVLAConfig, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config
        self.rtc_processor = rtc_processor

        self.vlm_with_expert = SmolVLMWithExpertModel(
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
        self.state_proj = nn.Linear(
            self.config.max_state_dim, self.vlm_with_expert.config.text_config.hidden_size
        )
        self.joint_motion_in_proj = nn.Linear( # TODO lets separate these to have latent in proj and action in proj
            2 * self.config.max_action_dim, self.vlm_with_expert.expert_hidden_size
        )
        self.joint_motion_out_proj = nn.Linear( # TODO lets separate these to have latent out proj and action out proj
            self.vlm_with_expert.expert_hidden_size, 2 * self.config.max_action_dim
        )
        self.joint_motion_time_mlp_in = nn.Linear( # TODO lets separate these to have latent time mlp in and action time mlp in
            self.vlm_with_expert.expert_hidden_size * 2, self.vlm_with_expert.expert_hidden_size
        )
        self.joint_motion_time_mlp_out = nn.Linear( # TODO lets separate these to have latent time mlp out and action time mlp out
            self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size
        )
        self.latent_step_dim = int(config.latent_vector_dim) // int(config.latent_code_seq_len)
        self.joint_motion_dim = 2 * int(config.max_action_dim)

        self.set_requires_grad()
        self.fake_image_token = self.vlm_with_expert.processor.tokenizer.fake_image_token_id
        self.global_image_token = self.vlm_with_expert.processor.tokenizer.global_image_token_id
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

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, state: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embs = []
        pad_masks = []
        att_masks = []

        for img, img_mask in zip(images, img_masks, strict=False):
            if self.add_image_special_tokens:
                image_start_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_start_mask = torch.ones_like(
                    image_start_token[:, :, 0], dtype=torch.bool, device=image_start_token.device
                )
                embs.append(image_start_token)
                pad_masks.append(image_start_mask)
                att_masks += [0] * image_start_mask.shape[-1]

            img_emb = self.vlm_with_expert.embed_image(img)
            img_emb = img_emb * torch.tensor(
                img_emb.shape[-1] ** 0.5, dtype=img_emb.dtype, device=img_emb.device
            )
            batch_size, num_img_embs = img_emb.shape[:2]
            expanded_img_mask = img_mask[:, None].expand(batch_size, num_img_embs)
            embs.append(img_emb)
            pad_masks.append(expanded_img_mask)
            att_masks += [0] * num_img_embs

            if self.add_image_special_tokens:
                image_end_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.image_end_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_end_mask = torch.ones_like(
                    image_end_token[:, :, 0], dtype=torch.bool, device=image_end_token.device
                )
                embs.append(image_end_token)
                pad_masks.append(image_end_mask)
                att_masks += [0] * image_end_mask.shape[1]

        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]

        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        batch_size = state_emb.shape[0]
        device = state_emb.device
        state_mask = torch.ones(batch_size, state_emb.shape[1], dtype=torch.bool, device=device)
        pad_masks.append(state_mask)
        att_masks += [1] * state_emb.shape[1]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)[None, :]

        seq_len = pad_masks.shape[1]
        if seq_len < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)

        return embs, pad_masks, att_masks.expand(batch_size, -1)

    def embed_suffix(
        self,
        noisy_motion: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if int(noisy_motion.shape[-1]) != int(self.joint_motion_dim):
            raise ValueError(
                "Expected packed joint motion suffix with shape [B, S, 2 * max_action_dim], "
                f"got {tuple(noisy_motion.shape)}"
            )
        motion_emb = self.joint_motion_in_proj(noisy_motion)
        device = motion_emb.device
        batch_size = motion_emb.shape[0]
        dtype = motion_emb.dtype
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        ).to(dtype=dtype)
        time_emb = time_emb[:, None, :].expand_as(motion_emb)
        motion_time_emb = torch.cat([motion_emb, time_emb], dim=2)
        motion_time_emb = self.joint_motion_time_mlp_in(motion_time_emb)
        motion_time_emb = F.silu(motion_time_emb)
        motion_time_emb = self.joint_motion_time_mlp_out(motion_time_emb)

        pad_masks = torch.ones(
            batch_size, motion_time_emb.shape[1], dtype=torch.bool, device=device
        )
        att_masks = torch.ones(
            batch_size, motion_time_emb.shape[1], dtype=motion_time_emb.dtype, device=device
        )
        return motion_time_emb, pad_masks, att_masks

    def _forward_suffix_hidden(
        self,
        *,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
        suffix_embs: torch.Tensor,
        suffix_pad_masks: torch.Tensor,
        suffix_att_masks: torch.Tensor,
    ) -> torch.Tensor:
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        return suffix_out.to(dtype=torch.float32)

    def _prepare_latent_targets(
        self,
        latent_vectors: torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if latent_vectors is None:
            return torch.zeros(
                batch_size,
                self.config.chunk_size,
                self.latent_step_dim,
                device=device,
                dtype=dtype,
            )
        target_seq = reshape_latent_vector_sequence(
            latent_vectors,
            latent_code_seq_len=int(self.config.latent_code_seq_len),
            latent_vector_dim=int(self.config.latent_vector_dim),
        ).to(device=device, dtype=dtype)
        target_seq = torch.nan_to_num(target_seq, nan=0.0, posinf=0.0, neginf=0.0)
        if int(target_seq.shape[1]) != int(self.config.chunk_size):
            raise ValueError(
                "Packed joint diffusion requires latent sequence length to match chunk_size, "
                f"got latent_seq_len={target_seq.shape[1]} chunk_size={self.config.chunk_size}"
            )
        if int(target_seq.shape[2]) > int(self.config.max_action_dim):
            raise ValueError(
                "Packed joint diffusion requires latent step dim <= max_action_dim, "
                f"got latent_step_dim={target_seq.shape[2]} max_action_dim={self.config.max_action_dim}"
            )
        return target_seq

    def _prepare_action_targets( # TODO remove all these helpers
        self,
        actions: torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if actions is None:
            return torch.zeros(
                batch_size,
                self.config.chunk_size,
                self.config.max_action_dim,
                device=device,
                dtype=dtype,
            )
        return actions.to(device=device, dtype=dtype)

    def _pack_joint_motion_targets(
        self,
        *,
        action_targets: torch.Tensor,
        latent_targets: torch.Tensor,
    ) -> torch.Tensor:
        latent_padded = (
            latent_targets
            if int(latent_targets.shape[-1]) == int(self.config.max_action_dim)
            else pad_vector(latent_targets, self.config.max_action_dim)
        )
        return torch.cat([latent_padded, action_targets], dim=-1)

    def _split_joint_motion(
        self,
        joint_motion: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent_motion = joint_motion[..., : self.config.max_action_dim]
        action_motion = joint_motion[..., self.config.max_action_dim :]
        return latent_motion, action_motion

    def _prepare_joint_noise(
        self,
        noise: torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        joint_shape = (batch_size, self.config.chunk_size, self.joint_motion_dim)
        if noise is None:
            return self.sample_noise(joint_shape, device)
        noise = noise.to(device=device, dtype=dtype)
        if tuple(noise.shape) == joint_shape:
            return noise
        action_shape = (batch_size, self.config.chunk_size, self.config.max_action_dim)
        if tuple(noise.shape) == action_shape:
            latent_noise = self.sample_noise(action_shape, device)
            return torch.cat([latent_noise.to(dtype=dtype), noise], dim=-1)
        raise ValueError(
            "Expected diffusion noise shaped like action targets or packed joint targets, "
            f"got {tuple(noise.shape)}"
        )

    def forward_joint_motion_losses(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor | None,
        actions: torch.Tensor | None,
        latent_vectors: torch.Tensor | None,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(state.shape[0])
        device = state.device
        dtype = torch.float32
        action_targets = self._prepare_action_targets( # TODO remove all these helpers
            actions,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        latent_targets = self._prepare_latent_targets(
            latent_vectors,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        joint_targets = self._pack_joint_motion_targets(
            action_targets=action_targets,
            latent_targets=latent_targets,
        )
        joint_noise = self._prepare_joint_noise(
            noise,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        if time is None:
            time = self.sample_time(batch_size, device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * joint_noise + (1 - time_expanded) * joint_targets
        u_t = joint_noise - joint_targets

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)
        suffix_out = self._forward_suffix_hidden( # TODO remove this helper function and put the code directly here, like in plain smolvla
            prefix_embs=prefix_embs,
            prefix_pad_masks=prefix_pad_masks,
            prefix_att_masks=prefix_att_masks,
            suffix_embs=suffix_embs,
            suffix_pad_masks=suffix_pad_masks,
            suffix_att_masks=suffix_att_masks,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :].to(dtype=torch.float32)
        v_t = self.joint_motion_out_proj(suffix_out)
        latent_u_t, action_u_t = self._split_joint_motion(u_t) # TODO check if we can simplify without helper
        latent_v_t, action_v_t = self._split_joint_motion(v_t)
        action_losses = F.mse_loss(action_u_t, action_v_t, reduction="none")
        latent_losses = F.mse_loss(latent_u_t, latent_v_t, reduction="none")
        return action_losses, latent_losses

    def forward( # TODO remove this helper function and put the code directly here, like in plain smolvla
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        actions,
        latent_vectors=None,
        noise=None,
        time=None,
    ) -> Tensor:
        action_losses, _ = self.forward_joint_motion_losses(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            actions,
            latent_vectors,
            noise=noise,
            time=time,
        )
        return action_losses

    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        batch_size = state.shape[0]
        device = state.device
        joint_noise = self._prepare_joint_noise(
            noise,
            batch_size=batch_size,
            device=device,
            dtype=torch.float32,
        )

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )
        num_steps = self.config.num_steps
        dt = -1.0 / num_steps

        x_t = joint_noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(batch_size)

            def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                return self.denoise_step(
                    x_t=input_x_t,
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    timestep=current_timestep,
                )

            if self._rtc_enabled():
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")
                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(x_t)

            x_t = x_t + dt * v_t

            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

        _, action_motion = self._split_joint_motion(x_t)
        return action_motion

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep)
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1][:, -self.config.chunk_size :].to(dtype=torch.float32)
        return self.joint_motion_out_proj(suffix_out)


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

    def prepare_images(self, batch):
        images = []
        img_masks = []
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )

        for key in present_img_keys:
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)
            img = img * 2.0 - 1.0

            batch_size = img.shape[0]
            device = img.device
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
            else:
                mask = torch.ones(batch_size, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)
        return images, img_masks

    def prepare_state(self, batch):
        state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        return pad_vector(state, self.config.max_state_dim)

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

    def _make_loss_metrics(
        self,
        *,
        branch_name: str,
        loss: Tensor,
        kept: int | float,
        denominator: int,
        kept_exact: int | float | None = None,
    ) -> dict[str, float | Tensor]:
        kept_value = float(kept)
        denominator_value = float(int(denominator))
        kept_exact_value = float(kept_exact if kept_exact is not None else kept)
        metrics: dict[str, float | Tensor] = {
            f"{branch_name}_loss": float(loss.detach().cpu()),
            f"{branch_name}_supervised_samples": kept_value,
            f"batch_{branch_name}_supervised_samples": kept_value,
            f"batch_{branch_name}_supervised_denominator": denominator_value,
            f"batch_{branch_name}_supervised_fraction": (
                kept_value / denominator_value if denominator_value > 0.0 else 0.0
            ),
            f"_{branch_name}_supervised_denominator": denominator_value,
            f"_{branch_name}_loss_denominator_exact": kept_exact_value,
            f"_{branch_name}_loss_tensor": loss,
        }
        return metrics

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
            latent_code_seq_len=int(self.config.latent_code_seq_len),
            latent_vector_dim=int(self.config.latent_vector_dim),
        ).to(dtype=torch.float32)
        sequence_length = int(latent_targets.shape[1])
        action_step_keep = action_keep.bool().reshape(batch_size, 1).expand(batch_size, sequence_length)
        if "action_is_pad" in batch:
            action_step_keep = action_step_keep & (~batch["action_is_pad"].to(device=labels.device, dtype=torch.bool))
        joint_step_keep = action_step_keep & latent_keep.bool()
        joint_count = int(joint_step_keep.any(dim=1).sum().item())
        joint_token_count = int(joint_step_keep.sum().item())
        metrics: dict[str, float] = {
            "batch_joint_supervised_samples": float(joint_count),
            "batch_joint_supervised_denominator": float(batch_size),
            "batch_joint_supervised_fraction": float(joint_count) / float(batch_size)
            if batch_size > 0
            else 0.0,
            "batch_joint_supervised_tokens": float(joint_token_count),
            "batch_joint_supervised_token_denominator": float(batch_size * sequence_length),
            "batch_joint_supervised_token_fraction": (
                float(joint_token_count) / float(batch_size * sequence_length)
                if batch_size > 0 and sequence_length > 0
                else 0.0
            ),
        }

        if ACTION not in batch:
            return metrics

        raw_actions = batch[ACTION].to(device=labels.device, dtype=torch.float32)
        metrics["action_target_rms"] = masked_rms(raw_actions, action_step_keep)
        metrics["action_target_abs_mean"] = masked_abs_mean(raw_actions, action_step_keep)
        metrics["latent_target_rms"] = masked_rms(latent_targets, latent_keep)
        metrics["latent_target_abs_mean"] = masked_abs_mean(latent_targets, latent_keep)

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
        joint_motion_prefixes = (
            "model.joint_motion_in_proj",
            "model.joint_motion_out_proj",
            "model.joint_motion_time_mlp_in",
            "model.joint_motion_time_mlp_out",
        )

        joint_motion_parameters: list[nn.Parameter] = []
        backbone_parameters: list[nn.Parameter] = []
        all_parameters: list[nn.Parameter] = []

        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            all_parameters.append(parameter)
            if name.startswith(joint_motion_prefixes):
                joint_motion_parameters.append(parameter)
            else:
                backbone_parameters.append(parameter)

        joint_motion_norm = parameter_grad_l2_norm(joint_motion_parameters)
        backbone_norm = parameter_grad_l2_norm(backbone_parameters)

        return {
            "grad_norm_model_total": parameter_grad_l2_norm(all_parameters),
            "grad_norm_shared": backbone_norm,
            "grad_norm_action_head": joint_motion_norm,
            "grad_norm_latent_head": joint_motion_norm,
            "grad_norm_backbone": backbone_norm,
            "grad_norm_joint_motion": joint_motion_norm,
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

    def _forward_joint_diffusion_branch(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None,
        time: Tensor | None,
    ) -> tuple[
        Tensor,
        dict[str, float | Tensor],
        Tensor,
        Tensor,
        dict[str, float | Tensor],
        Tensor,
        Tensor,
    ]:
        latent_key = str(self.config.latent_label_key)
        if self.config.training_mode in {"action", "multitask"} and ACTION not in batch:
            raise KeyError(f"Missing action batch key {ACTION!r}")
        if self.config.training_mode in {"latent", "multitask"} and latent_key not in batch:
            raise KeyError(f"Missing latent label batch key {latent_key!r}")

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        labels = (
            batch[latent_key].to(device=state.device, dtype=torch.float32)
            if latent_key in batch
            else None
        )
        actions = self.prepare_action(batch) if ACTION in batch else None
        action_is_pad = batch.get("action_is_pad")
        action_losses, latent_losses = self.model.forward_joint_motion_losses(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            actions,
            labels,
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
            max_action_dim=self.config.max_action_dim,
            action_is_pad=action_is_pad,
        )
        latent_is_pad = ~latent_keep
        if action_is_pad is not None:
            latent_is_pad = latent_is_pad | action_is_pad.to(device=latent_is_pad.device, dtype=torch.bool)
        per_sample_latent = reduce_action_per_sample(
            latent_losses,
            max_action_dim=self.config.max_action_dim,
            action_is_pad=latent_is_pad,
        )
        per_step_latent = latent_losses.mean(dim=-1)
        action_loss, action_kept = masked_mean_or_zero(per_sample_action, action_keep)
        latent_sample_keep = latent_keep.any(dim=1)
        latent_loss, _ = masked_mean_or_zero(per_sample_latent, latent_sample_keep)
        latent_kept_samples = int(latent_sample_keep.sum().item())
        latent_kept_tokens = int(latent_keep.sum().item())

        action_metrics = self._make_loss_metrics(
            branch_name="action",
            loss=action_loss,
            kept=action_kept,
            denominator=int(per_sample_action.shape[0]),
        )
        latent_metrics = self._make_loss_metrics(
            branch_name="latent",
            loss=latent_loss,
            kept=latent_kept_samples,
            denominator=int(per_sample_latent.shape[0]),
            kept_exact=latent_kept_samples,
        )
        latent_metrics["batch_latent_supervised_tokens"] = float(latent_kept_tokens)
        latent_metrics["batch_latent_supervised_token_denominator"] = float(
            int(per_step_latent.shape[0]) * int(per_step_latent.shape[1])
        )
        latent_metrics["batch_latent_supervised_token_fraction"] = (
            float(latent_kept_tokens)
            / float(int(per_step_latent.shape[0]) * int(per_step_latent.shape[1]))
            if int(per_step_latent.shape[0]) > 0 and int(per_step_latent.shape[1]) > 0
            else 0.0
        )
        action_metrics["joint_diffusion_forward"] = 1.0
        latent_metrics["joint_diffusion_forward"] = 1.0
        latent_metrics["latent_head_mode_joint_diffusion"] = 1.0
        return action_loss, action_metrics, action_keep, latent_loss, latent_metrics, latent_keep, labels

    def forward(
        self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"
    ) -> tuple[Tensor, dict]:
        if reduction != "mean":
            raise NotImplementedError("latent_smolvla currently supports reduction='mean' only")

        if self.config.adapt_to_pi_aloha and OBS_STATE in batch and ACTION in batch:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        metrics: dict[str, float | Tensor] = {}
        (
            action_loss,
            action_metrics,
            action_keep,
            latent_loss,
            latent_metrics,
            latent_keep,
            latent_vector_labels,
        ) = self._forward_joint_diffusion_branch(batch, noise, time)
        metrics.update(action_metrics)
        metrics.update(latent_metrics)
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
        metrics["latent_head_mode_joint_diffusion"] = 1.0
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

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise, **kwargs
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
        common_projections = [
            "state_proj",
            "joint_motion_in_proj",
            "joint_motion_out_proj",
            "joint_motion_time_mlp_in",
            "joint_motion_time_mlp_out",
        ]
        common_projection_pattern = "|".join(common_projections)
        target_modules = (
            rf"(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|model\.({common_projection_pattern}))"
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

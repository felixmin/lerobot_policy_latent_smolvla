from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.types import TransitionKey

from lerobot_policy_latent_smolvla.configuration_latent_smolvla import LatentSmolVLAConfig
from lerobot_policy_latent_smolvla.loss_utils import (
    expand_keep_mask,
    make_sample_keep_mask,
    make_sequence_keep_mask,
    masked_mean_or_zero,
    pool_hidden,
    reshape_latent_vector_sequence,
)
from lerobot_policy_latent_smolvla.modeling_latent_smolvla import LatentSmolVLAFlowMatching, LatentSmolVLAPolicy
from lerobot_policy_latent_smolvla.modeling_latent_smolvla import (
    masked_abs_mean,
    masked_rms,
    parameter_grad_l2_norm,
)
from lerobot_policy_latent_smolvla.processor_latent_smolvla import (
    LatentSmolVLALatentTargetNormalizer,
    _make_batch_to_transition_with_latent_keys,
)


def test_config_defaults():
    config = LatentSmolVLAConfig()
    assert config.training_mode == "multitask"
    assert config.latent_head_mode == "vector_diffusion"
    assert config.latent_label_key == "latent_labels.continuous_vector_latents"
    assert config.latent_valid_key == "latent_labels.valid"
    assert config.normalize_latent_targets is True
    assert config.latent_normalization_source == "latent"
    assert config.latent_supervision_key is None
    assert config.action_supervision_key is None
    assert config.latent_code_seq_len == config.chunk_size
    assert not isinstance(config, SmolVLAConfig)
    assert isinstance(config.get_optimizer_preset(), AdamWConfig)
    assert isinstance(config.get_scheduler_preset(), CosineDecayWithWarmupSchedulerConfig)


def test_config_rejects_invalid_latent_normalization_source():
    with pytest.raises(ValueError, match="latent_normalization_source"):
        LatentSmolVLAConfig(latent_normalization_source="prefer_action")


def test_config_allows_shorter_latent_horizon_than_action_horizon():
    config = LatentSmolVLAConfig(
        chunk_size=6,
        n_action_steps=6,
        latent_code_seq_len=4,
        latent_vector_dim=24,
    )
    assert config.chunk_size == 6
    assert config.latent_code_seq_len == 4


def test_config_rejects_mismatched_latent_delta_indices_length():
    with pytest.raises(ValueError, match="latent_delta_indices must have length latent_code_seq_len"):
        LatentSmolVLAConfig(
            chunk_size=6,
            n_action_steps=6,
            latent_code_seq_len=4,
            latent_vector_dim=24,
            latent_delta_indices=[0, 10, 20],
        )


def test_config_requires_latent_step_dim_to_fit_latent_pad_dim():
    with pytest.raises(ValueError, match="latent step dim <= max_latent_dim"):
        LatentSmolVLAConfig(
            chunk_size=4,
            n_action_steps=4,
            latent_code_seq_len=4,
            latent_vector_dim=28,
            max_latent_dim=6,
        )


def test_config_allows_latent_step_dim_to_exceed_action_pad_dim_when_latent_pad_dim_is_larger():
    config = LatentSmolVLAConfig(
        chunk_size=4,
        n_action_steps=4,
        latent_code_seq_len=4,
        latent_vector_dim=28,
        max_action_dim=6,
        max_latent_dim=8,
    )
    assert config.max_action_dim == 6
    assert config.max_latent_dim == 8


def test_preserve_configured_latent_and_supervision_keys():
    config = LatentSmolVLAConfig(
        latent_code_seq_len=4,
        latent_vector_dim=16,
        chunk_size=4,
        n_action_steps=4,
        latent_supervision_key="latent_supervision",
        action_supervision_key="action_supervision",
    )
    to_transition = _make_batch_to_transition_with_latent_keys(config)
    batch = {
        config.latent_label_key: torch.randn(2, 4, 4),
        config.latent_valid_key: torch.tensor([True, False]),
        config.latent_supervision_key: torch.tensor([True, True]),
        config.action_supervision_key: torch.tensor([True, False]),
    }

    transition = to_transition(batch)
    complementary_data = transition[TransitionKey.COMPLEMENTARY_DATA]

    assert torch.equal(complementary_data[config.latent_label_key], batch[config.latent_label_key])
    assert torch.equal(complementary_data[config.latent_valid_key], batch[config.latent_valid_key])
    assert torch.equal(
        complementary_data[config.latent_supervision_key],
        batch[config.latent_supervision_key],
    )
    assert torch.equal(
        complementary_data[config.action_supervision_key],
        batch[config.action_supervision_key],
    )


def test_latent_target_normalizer_applies_mean_std_to_complementary_data():
    step = LatentSmolVLALatentTargetNormalizer(
        latent_label_key="latent_labels.continuous_vector_latents",
        stats={
            "mean": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "std": torch.tensor([[2.0, 4.0], [5.0, 10.0]]),
        },
    )
    transition = {
        TransitionKey.COMPLEMENTARY_DATA: {
            "latent_labels.continuous_vector_latents": torch.tensor(
                [[[3.0, 6.0], [8.0, 14.0]]],
                dtype=torch.float32,
            )
        }
    }

    transformed = step(transition)
    expected = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]], dtype=torch.float32)
    assert torch.allclose(
        transformed[TransitionKey.COMPLEMENTARY_DATA]["latent_labels.continuous_vector_latents"],
        expected,
    )


def test_latent_target_normalizer_supports_flat_labels_with_structured_stats():
    step = LatentSmolVLALatentTargetNormalizer(
        latent_label_key="latent_labels.continuous_vector_latents",
        stats={
            "mean": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "std": torch.tensor([[2.0, 4.0], [5.0, 10.0]]),
        },
    )
    transition = {
        TransitionKey.COMPLEMENTARY_DATA: {
            "latent_labels.continuous_vector_latents": torch.tensor(
                [[3.0, 6.0, 8.0, 14.0]],
                dtype=torch.float32,
            )
        }
    }

    transformed = step(transition)
    expected = torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
    assert torch.allclose(
        transformed[TransitionKey.COMPLEMENTARY_DATA]["latent_labels.continuous_vector_latents"],
        expected,
    )


def test_latent_target_normalizer_defaults_to_latent_stats():
    step = LatentSmolVLALatentTargetNormalizer(
        latent_label_key="latent_labels.continuous_vector_latents",
        stats={
            "mean": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "std": torch.tensor([[2.0, 4.0], [5.0, 10.0]]),
        },
        action_stats={
            "mean": torch.tensor([100.0, 200.0]),
            "std": torch.tensor([10.0, 10.0]),
        },
    )
    transition = {
        TransitionKey.COMPLEMENTARY_DATA: {
            "latent_labels.continuous_vector_latents": torch.tensor(
                [[[3.0, 6.0], [8.0, 14.0]]],
                dtype=torch.float32,
            )
        }
    }

    transformed = step(transition)
    expected = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]], dtype=torch.float32)
    assert torch.allclose(
        transformed[TransitionKey.COMPLEMENTARY_DATA]["latent_labels.continuous_vector_latents"],
        expected,
    )


def test_latent_target_normalizer_can_use_action_stats_for_action_like_labels():
    step = LatentSmolVLALatentTargetNormalizer(
        latent_label_key="latent_labels.continuous_vector_latents",
        normalization_source="action",
        stats={
            "mean": torch.tensor([[100.0, 200.0], [300.0, 400.0]]),
            "std": torch.tensor([[10.0, 10.0], [10.0, 10.0]]),
        },
        action_stats={
            "mean": torch.tensor([1.0, 2.0]),
            "std": torch.tensor([2.0, 4.0]),
        },
    )
    transition = {
        TransitionKey.COMPLEMENTARY_DATA: {
            "latent_labels.continuous_vector_latents": torch.tensor(
                [[[3.0, 6.0], [5.0, 10.0]]],
                dtype=torch.float32,
            )
        }
    }

    transformed = step(transition)
    expected = torch.tensor([[[1.0, 1.0], [2.0, 2.0]]], dtype=torch.float32)
    assert torch.allclose(
        transformed[TransitionKey.COMPLEMENTARY_DATA]["latent_labels.continuous_vector_latents"],
        expected,
    )


def test_latent_target_normalizer_action_source_raises_without_action_stats():
    step = LatentSmolVLALatentTargetNormalizer(
        latent_label_key="latent_labels.continuous_vector_latents",
        normalization_source="action",
        stats={
            "mean": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "std": torch.tensor([[2.0, 4.0], [5.0, 10.0]]),
        },
        action_stats=None,
    )
    transition = {
        TransitionKey.COMPLEMENTARY_DATA: {
            "latent_labels.continuous_vector_latents": torch.ones(1, 2, 2, dtype=torch.float32)
        }
    }

    with pytest.raises(ValueError, match="action stats are missing"):
        step(transition)


def test_latent_target_normalizer_raises_when_enabled_without_stats():
    step = LatentSmolVLALatentTargetNormalizer(
        latent_label_key="latent_labels.continuous_vector_latents",
        enabled=True,
        stats=None,
    )
    transition = {
        TransitionKey.COMPLEMENTARY_DATA: {
            "latent_labels.continuous_vector_latents": torch.ones(1, 2, 2, dtype=torch.float32)
        }
    }

    with pytest.raises(ValueError, match="stats are missing"):
        step(transition)


def test_pool_hidden_masked_mean():
    hidden = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    mask = torch.tensor([[True, True, False], [False, True, True]])
    pooled = pool_hidden(hidden, mask)
    expected0 = hidden[0, :2].mean(dim=0)
    expected1 = hidden[1, 1:].mean(dim=0)
    assert torch.allclose(pooled[0], expected0)
    assert torch.allclose(pooled[1], expected1)


def test_make_sample_keep_mask_defaults_to_all_true():
    keep = make_sample_keep_mask({}, key=None, batch_size=3, device=torch.device("cpu"))
    assert torch.equal(keep, torch.tensor([True, True, True]))


def test_make_sequence_keep_mask_expands_sample_level_masks():
    keep = make_sequence_keep_mask(
        {"latent_labels.valid": torch.tensor([True, False])},
        key="latent_labels.valid",
        batch_size=2,
        sequence_length=4,
        device=torch.device("cpu"),
    )
    assert torch.equal(
        keep,
        torch.tensor(
            [[True, True, True, True], [False, False, False, False]],
            dtype=torch.bool,
        ),
    )


def test_make_sequence_keep_mask_preserves_step_level_masks():
    keep = make_sequence_keep_mask(
        {"latent_labels.valid": torch.tensor([[True, False, True], [False, True, False]])},
        key="latent_labels.valid",
        batch_size=2,
        sequence_length=3,
        device=torch.device("cpu"),
    )
    assert torch.equal(
        keep,
        torch.tensor([[True, False, True], [False, True, False]], dtype=torch.bool),
    )


def test_expand_keep_mask_supports_sequence_masks():
    values = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)
    keep = torch.tensor([[True, False, True], [False, True, False]], dtype=torch.bool)
    expanded = expand_keep_mask(values, keep)
    assert expanded.shape == values.shape
    assert expanded[0, 0].all()
    assert not expanded[0, 1].any()
    assert expanded[1, 1].all()


def test_masked_mean_or_zero_supports_sequence_masks():
    values = torch.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    keep = torch.tensor([[True, False, True], [False, False, False]])
    mean, kept = masked_mean_or_zero(values, keep)
    assert kept == 2
    assert float(mean) == pytest.approx(2.0)


def test_masked_target_scale_helpers():
    values = torch.tensor(
        [
            [[3.0, 4.0], [0.0, 0.0]],
            [[10.0, 0.0], [0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    keep = torch.tensor([True, False])

    assert masked_rms(values, keep) == 2.5
    assert masked_abs_mean(values, keep) == 1.75


def test_masked_target_scale_helpers_support_sequence_masks():
    values = torch.tensor(
        [
            [[3.0, 4.0], [0.0, 0.0]],
            [[10.0, 0.0], [0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    keep = torch.tensor([[True, False], [False, False]])

    assert masked_rms(values, keep) == pytest.approx((25.0 / 2.0) ** 0.5)
    assert masked_abs_mean(values, keep) == pytest.approx(3.5)


def test_policy_latent_keep_mask_supports_sequence_validity():
    policy = LatentSmolVLAPolicy.__new__(LatentSmolVLAPolicy)
    policy.config = SimpleNamespace(
        latent_valid_key="latent_labels.valid",
        latent_supervision_key="latent_supervision",
    )
    keep = policy._make_latent_keep_mask(
        {
            "latent_labels.valid": torch.tensor([[True, False, True], [False, True, True]]),
            "latent_supervision": torch.tensor([True, False]),
        },
        batch_size=2,
        sequence_length=3,
        device=torch.device("cpu"),
    )
    assert torch.equal(
        keep,
        torch.tensor(
            [[True, False, True], [False, False, False]],
            dtype=torch.bool,
        ),
    )


def test_policy_latent_keep_mask_intersects_valid_samples():
    policy = LatentSmolVLAPolicy.__new__(LatentSmolVLAPolicy)
    policy.config = SimpleNamespace(
        latent_valid_key="latent_labels.valid",
        latent_supervision_key=None,
    )
    keep = policy._make_latent_keep_mask(
        {
            "latent_labels.valid": torch.tensor([[True, True, True], [True, True, True]]),
        },
        batch_size=2,
        sequence_length=3,
        device=torch.device("cpu"),
        valid_samples=torch.tensor([[True, False, True], [False, True, False]]),
    )
    assert torch.equal(
        keep,
        torch.tensor(
            [[True, False, True], [False, True, False]],
            dtype=torch.bool,
        ),
    )


def test_parameter_grad_l2_norm_ignores_missing_grads():
    first = nn.Parameter(torch.zeros(2, dtype=torch.float32))
    second = nn.Parameter(torch.zeros(2, dtype=torch.float32))
    first.grad = torch.tensor([3.0, 4.0], dtype=torch.float32)
    second.grad = None

    assert parameter_grad_l2_norm([first, second]) == 5.0


def test_reshape_latent_vector_sequence_from_flat():
    vectors = torch.arange(24, dtype=torch.float32).reshape(2, 12)
    seq = reshape_latent_vector_sequence(
        vectors,
        latent_code_seq_len=3,
        latent_vector_dim=12,
    )
    assert seq.shape == (2, 3, 4)
    assert torch.equal(seq.reshape(2, 12), vectors)


def test_policy_default_peft_targets_use_vector_diffusion_modules():
    policy = LatentSmolVLAPolicy.__new__(LatentSmolVLAPolicy)

    targets = policy._get_default_peft_targets()

    assert "latent_in_proj" in targets["target_modules"]
    assert "latent_plan_proj" in targets["target_modules"]
    assert "action_in_proj" in targets["target_modules"]
    assert "action_out_proj" in targets["target_modules"]
    assert targets["modules_to_save"] == []


def test_model_forward_returns_unreduced_action_and_latent_losses_shape():
    model = LatentSmolVLAFlowMatching.__new__(LatentSmolVLAFlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        chunk_size=4,
        max_action_dim=6,
        max_latent_dim=6,
        latent_code_seq_len=4,
        latent_vector_dim=16,
        latent_teacher_force_ratio_end=0.0,
    )
    model.latent_step_dim = 4
    model.max_latent_dim = 6
    model.latent_out_proj = nn.Linear(7, 6)
    model.action_out_proj = nn.Linear(7, 6)
    model.sample_noise = lambda shape, device: torch.zeros(shape, device=device)
    model.sample_time = lambda batch_size, device: torch.full((batch_size,), 0.5, device=device)
    model.embed_prefix = lambda *args, **kwargs: (
        torch.zeros(2, 3, 7),
        torch.ones(2, 3, dtype=torch.bool),
        torch.zeros(2, 3, dtype=torch.bool),
    )
    model.embed_latent_suffix = lambda noisy_motion, timestep: (
        torch.zeros(noisy_motion.shape[0], noisy_motion.shape[1], 7),
        torch.ones(noisy_motion.shape[0], noisy_motion.shape[1], dtype=torch.bool),
        torch.ones(noisy_motion.shape[0], noisy_motion.shape[1]),
    )
    model.embed_action_suffix = lambda noisy_motion, timestep: (
        torch.zeros(noisy_motion.shape[0], noisy_motion.shape[1], 7),
        torch.ones(noisy_motion.shape[0], noisy_motion.shape[1], dtype=torch.bool),
        torch.ones(noisy_motion.shape[0], noisy_motion.shape[1]),
    )
    model.embed_latent_plan = lambda latent_plan, latent_valid=None, action_horizon=4: (
        torch.zeros(latent_plan.shape[0], latent_plan.shape[1], 7),
        torch.ones(latent_plan.shape[0], latent_plan.shape[1], dtype=torch.bool),
        torch.zeros(latent_plan.shape[0], latent_plan.shape[1], dtype=torch.bool),
    )
    model.latent_vlm_with_expert = SimpleNamespace(forward=lambda **kwargs: ([kwargs["inputs_embeds"][0], torch.ones_like(kwargs["inputs_embeds"][1])], None))
    model.action_vlm_with_expert = SimpleNamespace(forward=lambda **kwargs: ([kwargs["inputs_embeds"][0], torch.ones_like(kwargs["inputs_embeds"][1])], None))
    model.eval()

    actions = torch.randn(2, 4, 6)
    latent_vectors = torch.randn(2, 4, 4)
    action_losses, latent_losses = model.forward(
        None,
        None,
        None,
        None,
        torch.zeros(2, 6),
        actions,
        latent_vectors=latent_vectors,
        noise=(torch.zeros(2, 4, 6), torch.zeros(2, 4, 6)),
        time=torch.full((2,), 0.5),
    )
    assert action_losses.shape == actions.shape
    assert latent_losses.shape == actions.shape


def test_model_forward_returns_action_and_latent_shapes():
    model = LatentSmolVLAFlowMatching.__new__(LatentSmolVLAFlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        chunk_size=4,
        max_action_dim=6,
        max_latent_dim=6,
        latent_code_seq_len=4,
        latent_vector_dim=16,
        latent_teacher_force_ratio_end=0.0,
    )
    model.latent_step_dim = 4
    model.max_latent_dim = 6
    model.latent_out_proj = nn.Linear(7, 6)
    model.action_out_proj = nn.Linear(7, 6)
    model.sample_time = lambda batch_size, device: torch.full((batch_size,), 0.5, device=device)
    model.sample_noise = lambda shape, device: torch.zeros(shape, device=device)
    model.embed_prefix = lambda *args, **kwargs: (
        torch.zeros(2, 4, 7),
        torch.ones(2, 4, dtype=torch.bool),
        torch.zeros(2, 4, dtype=torch.bool),
    )
    model.embed_latent_suffix = lambda noisy_motion, timestep: (
        torch.zeros(noisy_motion.shape[0], noisy_motion.shape[1], 7),
        torch.ones(noisy_motion.shape[0], noisy_motion.shape[1], dtype=torch.bool),
        torch.ones(noisy_motion.shape[0], noisy_motion.shape[1]),
    )
    model.embed_action_suffix = lambda noisy_motion, timestep: (
        torch.zeros(noisy_motion.shape[0], noisy_motion.shape[1], 7),
        torch.ones(noisy_motion.shape[0], noisy_motion.shape[1], dtype=torch.bool),
        torch.ones(noisy_motion.shape[0], noisy_motion.shape[1]),
    )
    model.embed_latent_plan = lambda latent_plan, latent_valid=None, action_horizon=4: (
        torch.zeros(latent_plan.shape[0], latent_plan.shape[1], 7),
        torch.ones(latent_plan.shape[0], latent_plan.shape[1], dtype=torch.bool),
        torch.zeros(latent_plan.shape[0], latent_plan.shape[1], dtype=torch.bool),
    )
    model.latent_vlm_with_expert = SimpleNamespace(forward=lambda **kwargs: ([kwargs["inputs_embeds"][0], torch.ones_like(kwargs["inputs_embeds"][1])], None))
    model.action_vlm_with_expert = SimpleNamespace(forward=lambda **kwargs: ([kwargs["inputs_embeds"][0], torch.ones_like(kwargs["inputs_embeds"][1])], None))
    model.eval()

    actions = torch.randn(2, 4, 6)
    latents = torch.randn(2, 4, 4)
    action_losses, latent_losses = model.forward(
        None,
        None,
        None,
        None,
        torch.zeros(2, 6),
        actions,
        latents,
        noise=(torch.zeros(2, 4, 6), torch.zeros(2, 4, 6)),
        time=torch.full((2,), 0.5),
    )

    assert action_losses.shape == actions.shape
    assert latent_losses.shape == (2, 4, 6)


def test_model_forward_rejects_mismatched_latent_horizon():
    model = LatentSmolVLAFlowMatching.__new__(LatentSmolVLAFlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        chunk_size=4,
        max_action_dim=6,
        max_latent_dim=6,
        latent_code_seq_len=4,
        latent_vector_dim=16,
        latent_teacher_force_ratio_end=0.0,
    )
    model.latent_step_dim = 4
    model.max_latent_dim = 6
    model.latent_out_proj = nn.Linear(7, 6)
    model.action_out_proj = nn.Linear(7, 6)
    model.sample_time = lambda batch_size, device: torch.full((batch_size,), 0.5, device=device)
    model.sample_noise = lambda shape, device: torch.zeros(shape, device=device)
    model.embed_prefix = lambda *args, **kwargs: (
        torch.zeros(2, 4, 7),
        torch.ones(2, 4, dtype=torch.bool),
        torch.zeros(2, 4, dtype=torch.bool),
    )
    model.embed_latent_suffix = lambda noisy_motion, timestep: (
        torch.zeros(noisy_motion.shape[0], noisy_motion.shape[1], 7),
        torch.ones(noisy_motion.shape[0], noisy_motion.shape[1], dtype=torch.bool),
        torch.ones(noisy_motion.shape[0], noisy_motion.shape[1]),
    )
    model.embed_action_suffix = lambda noisy_motion, timestep: (
        torch.zeros(noisy_motion.shape[0], noisy_motion.shape[1], 7),
        torch.ones(noisy_motion.shape[0], noisy_motion.shape[1], dtype=torch.bool),
        torch.ones(noisy_motion.shape[0], noisy_motion.shape[1]),
    )
    model.embed_latent_plan = lambda latent_plan, latent_valid=None, action_horizon=4: (
        torch.zeros(latent_plan.shape[0], latent_plan.shape[1], 7),
        torch.ones(latent_plan.shape[0], latent_plan.shape[1], dtype=torch.bool),
        torch.zeros(latent_plan.shape[0], latent_plan.shape[1], dtype=torch.bool),
    )
    model.latent_vlm_with_expert = SimpleNamespace(forward=lambda **kwargs: ([kwargs["inputs_embeds"][0], torch.ones_like(kwargs["inputs_embeds"][1])], None))
    model.action_vlm_with_expert = SimpleNamespace(forward=lambda **kwargs: ([kwargs["inputs_embeds"][0], torch.ones_like(kwargs["inputs_embeds"][1])], None))
    model.eval()

    with pytest.raises(ValueError, match=r"Expected latent vector tensor \[B,4,4\]"):
        model.forward(
            None,
            None,
            None,
            None,
            torch.zeros(2, 6),
            torch.randn(2, 4, 6),
            torch.randn(2, 3, 4),
            noise=(torch.zeros(2, 4, 6), torch.zeros(2, 4, 6)),
            time=torch.full((2,), 0.5),
        )


def test_policy_forward_combines_action_and_latent_losses():
    class DummyModel:
        def __call__(self, *args, **kwargs):
            return (
                torch.full((2, 4, 6), 2.0, dtype=torch.float32),
                torch.full((2, 4, 6), 3.0, dtype=torch.float32),
            )

    policy = LatentSmolVLAPolicy.__new__(LatentSmolVLAPolicy)
    policy.config = SimpleNamespace(
        adapt_to_pi_aloha=False,
        training_mode="multitask",
        action_loss_weight=1.0,
        latent_loss_weight=1.0,
        latent_label_key="latent_labels.continuous_vector_latents",
        latent_valid_key="latent_labels.valid",
        latent_supervision_key=None,
        action_supervision_key=None,
        max_action_dim=6,
    )
    policy._compute_joint_vector_target_metrics = lambda *args, **kwargs: {}
    policy.prepare_images = lambda batch: ([torch.zeros(2, 3, 4, 4)], [torch.ones(2, dtype=torch.bool)])
    policy.prepare_state = lambda batch: torch.zeros(2, 6)
    policy.model = DummyModel()

    total, metrics = policy.forward(
        {
            "action": torch.zeros(2, 4, 6),
            "latent_labels.continuous_vector_latents": torch.zeros(2, 4, 4),
            "latent_labels.valid": torch.ones(2, 4, dtype=torch.bool),
            "observation.language.tokens": torch.zeros(2, 5, dtype=torch.long),
            "observation.language.attention_mask": torch.ones(2, 5, dtype=torch.bool),
        }
    )

    assert float(total) == 5.0
    assert metrics["action_loss_weighted"] == 2.0
    assert metrics["latent_loss_weighted"] == 3.0
    assert metrics["action_loss"] == 2.0
    assert metrics["latent_loss"] == 3.0


def test_policy_forward_aligns_action_and_latent_reduction():
    class DummyModel:
        def __call__(self, *args, **kwargs):
            return (
                torch.ones(2, 4, 6, dtype=torch.float32),
                torch.ones(2, 4, 6, dtype=torch.float32),
            )

    policy = LatentSmolVLAPolicy.__new__(LatentSmolVLAPolicy)
    policy.config = SimpleNamespace(
        adapt_to_pi_aloha=False,
        training_mode="multitask",
        action_loss_weight=1.0,
        latent_loss_weight=1.0,
        latent_label_key="latent_labels.continuous_vector_latents",
        latent_valid_key="latent_labels.valid",
        latent_supervision_key=None,
        action_supervision_key=None,
        max_action_dim=6,
    )
    policy.prepare_images = lambda batch: ([torch.zeros(2, 3, 4, 4)], [torch.ones(2, dtype=torch.bool)])
    policy.prepare_state = lambda batch: torch.zeros(2, 6)
    policy.model = DummyModel()

    batch = {
        "action": torch.randn(2, 4, 6),
        "action_is_pad": torch.tensor([[False, False, True, True], [False, False, False, True]]),
        "latent_labels.continuous_vector_latents": torch.randn(2, 4, 4),
        "latent_labels.valid": torch.tensor([[True, True, False, False], [True, True, True, False]]),
        "observation.language.tokens": torch.zeros(2, 5, dtype=torch.long),
        "observation.language.attention_mask": torch.ones(2, 5, dtype=torch.bool),
    }

    policy._compute_joint_vector_target_metrics = lambda *args, **kwargs: {}

    total, metrics = policy.forward(batch, noise=None, time=None)

    assert float(total) == pytest.approx(1.25)
    assert metrics["action_loss"] == pytest.approx(0.625)
    assert metrics["latent_loss"] == pytest.approx(0.625)
    assert metrics["batch_latent_supervised_tokens"] == 5.0


def test_policy_forward_combines_latent_validity_with_dataset_padding():
    class DummyModel:
        def __call__(self, *args, **kwargs):
            self.kwargs = kwargs
            return (
                torch.ones(2, 4, 6, dtype=torch.float32),
                torch.ones(2, 4, 6, dtype=torch.float32),
            )

    policy = LatentSmolVLAPolicy.__new__(LatentSmolVLAPolicy)
    policy.config = SimpleNamespace(
        adapt_to_pi_aloha=False,
        training_mode="multitask",
        action_loss_weight=1.0,
        latent_loss_weight=1.0,
        latent_label_key="latent_labels.continuous_vector_latents",
        latent_valid_key="latent_labels.valid",
        latent_supervision_key=None,
        action_supervision_key=None,
        max_action_dim=6,
        latent_code_seq_len=4,
    )
    policy._compute_joint_vector_target_metrics = lambda *args, **kwargs: {}
    policy.prepare_images = lambda batch: ([torch.zeros(2, 3, 4, 4)], [torch.ones(2, dtype=torch.bool)])
    policy.prepare_state = lambda batch: torch.zeros(2, 6)
    policy.model = DummyModel()

    batch = {
        "action": torch.randn(2, 4, 6),
        "latent_labels.continuous_vector_latents": torch.randn(2, 4, 4),
        "latent_labels.continuous_vector_latents_is_pad": torch.tensor(
            [[False, True, False, False], [False, False, True, True]]
        ),
        "latent_labels.valid": torch.tensor([[True, True, True, False], [True, True, True, True]]),
        "observation.language.tokens": torch.zeros(2, 5, dtype=torch.long),
        "observation.language.attention_mask": torch.ones(2, 5, dtype=torch.bool),
    }

    total, metrics = policy.forward(batch)

    expected_keep = torch.tensor(
        [[True, False, True, False], [True, True, False, False]],
        dtype=torch.bool,
    )
    assert torch.equal(policy.model.kwargs["latent_valid"], expected_keep)
    assert float(total) == pytest.approx(1.5)
    assert metrics["latent_loss"] == pytest.approx(0.5)
    assert metrics["batch_latent_supervised_tokens"] == 4.0

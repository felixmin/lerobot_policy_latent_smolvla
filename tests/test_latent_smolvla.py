import torch

from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.types import TransitionKey

from lerobot_policy_latent_smolvla.configuration_latent_smolvla import LatentSmolVLAConfig
from lerobot_policy_latent_smolvla.loss_utils import (
    make_noisy_target,
    make_sample_keep_mask,
    pool_hidden,
    reduce_latent_per_sample,
    reduce_vector_flow_per_sample,
    reshape_latent_vector_sequence,
)
from lerobot_policy_latent_smolvla.processor_latent_smolvla import (
    _make_batch_to_transition_with_latent_keys,
)


def test_config_defaults():
    config = LatentSmolVLAConfig()
    assert config.training_mode == "multitask"
    assert config.latent_head_mode == "vector_diffusion"
    assert config.latent_label_key == "latent_labels.continuous_vector_latents"
    assert config.latent_valid_key == "latent_labels.valid"
    assert config.latent_supervision_key is None
    assert config.action_supervision_key is None
    assert not isinstance(config, SmolVLAConfig)
    assert isinstance(config.get_optimizer_preset(), AdamWConfig)
    assert isinstance(config.get_scheduler_preset(), CosineDecayWithWarmupSchedulerConfig)


def test_preserve_configured_latent_and_supervision_keys():
    config = LatentSmolVLAConfig(
        latent_supervision_key="latent_supervision",
        action_supervision_key="action_supervision",
    )
    to_transition = _make_batch_to_transition_with_latent_keys(config)
    batch = {
        config.latent_label_key: torch.randn(2, 12),
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


def test_reduce_latent_per_sample_ignores_missing_labels():
    logits = torch.tensor(
        [
            [[8.0, 0.0, 0.0], [0.0, 7.0, 0.0]],
            [[0.0, 0.0, 9.0], [0.0, 0.0, 0.0]],
        ]
    )
    labels = torch.tensor(
        [
            [0, 1],
            [2, -100],
        ]
    )
    per_sample_loss, valid_samples, per_sample_acc, per_sample_conf = reduce_latent_per_sample(
        logits,
        labels,
        ignore_index=-100,
    )
    assert torch.equal(valid_samples, torch.tensor([True, True]))
    assert torch.all(per_sample_loss >= 0)
    assert torch.allclose(per_sample_acc, torch.ones_like(per_sample_acc))
    assert torch.all(per_sample_conf > 0.5)


def test_reduce_latent_per_sample_accepts_scalar_labels_for_seq_len_one():
    logits = torch.tensor([[[8.0, 0.0, 0.0]], [[0.0, 7.0, 0.0]]])
    labels = torch.tensor([0, 1])
    per_sample_loss, valid_samples, per_sample_acc, _ = reduce_latent_per_sample(
        logits,
        labels,
        ignore_index=-100,
    )
    assert torch.equal(valid_samples, torch.tensor([True, True]))
    assert torch.all(per_sample_loss >= 0)
    assert torch.allclose(per_sample_acc, torch.ones_like(per_sample_acc))


def test_reshape_latent_vector_sequence_from_flat():
    vectors = torch.arange(24, dtype=torch.float32).reshape(2, 12)
    seq = reshape_latent_vector_sequence(
        vectors,
        latent_code_seq_len=3,
        latent_vector_dim=12,
    )
    assert seq.shape == (2, 3, 4)
    assert torch.equal(seq.reshape(2, 12), vectors)


def test_make_noisy_target_shapes():
    target = torch.ones(2, 3, 4)
    noise = torch.zeros_like(target)
    time = torch.tensor([0.25, 0.75])
    x_t, u_t = make_noisy_target(target=target, noise=noise, time=time)
    assert x_t.shape == target.shape
    assert u_t.shape == target.shape
    assert torch.allclose(u_t, -torch.ones_like(target))


def test_reduce_vector_flow_per_sample_zero_when_equal():
    target = torch.randn(2, 3, 4)
    per_sample = reduce_vector_flow_per_sample(target, target)
    assert torch.allclose(per_sample, torch.zeros_like(per_sample))

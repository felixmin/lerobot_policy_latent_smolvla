from types import SimpleNamespace

import torch
from torch import nn

from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.types import TransitionKey

from lerobot_policy_latent_smolvla.configuration_latent_smolvla import LatentSmolVLAConfig
from lerobot_policy_latent_smolvla.loss_utils import (
    make_sample_keep_mask,
    pool_hidden,
    reduce_latent_per_sample,
    reduce_vector_flow_per_sample,
    reshape_latent_vector_sequence,
)
from lerobot_policy_latent_smolvla.modeling_latent_smolvla import LatentSmolVLAFlowMatching, LatentSmolVLAPolicy
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
    assert config.latent_supervision_key is None
    assert config.action_supervision_key is None
    assert not isinstance(config, SmolVLAConfig)
    assert isinstance(config.get_optimizer_preset(), AdamWConfig)
    assert isinstance(config.get_scheduler_preset(), CosineDecayWithWarmupSchedulerConfig)


def test_config_accepts_vector_mse_latent_head_mode():
    config = LatentSmolVLAConfig(latent_head_mode="vector_mse")
    assert config.latent_head_mode == "vector_mse"


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

    try:
        step(transition)
    except ValueError as exc:
        assert "stats are missing" in str(exc)
    else:
        raise AssertionError("Expected latent target normalization to require stats when enabled.")


def test_latent_target_normalizer_leaves_discrete_labels_unchanged():
    step = LatentSmolVLALatentTargetNormalizer(
        latent_label_key="latent_labels.codebook_id_latents",
        enabled=True,
        stats=None,
    )
    labels = torch.tensor([[1, 2, 3]], dtype=torch.int64)
    transition = {
        TransitionKey.COMPLEMENTARY_DATA: {
            "latent_labels.codebook_id_latents": labels,
        }
    }

    transformed = step(transition)
    assert torch.equal(
        transformed[TransitionKey.COMPLEMENTARY_DATA]["latent_labels.codebook_id_latents"],
        labels,
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


def test_inline_noisy_target_shapes():
    target = torch.ones(2, 3, 4)
    noise = torch.zeros_like(target)
    time = torch.tensor([0.25, 0.75])
    time_expanded = time[:, None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * target
    u_t = noise - target
    assert x_t.shape == target.shape
    assert u_t.shape == target.shape
    assert torch.allclose(u_t, -torch.ones_like(target))


def test_reduce_vector_flow_per_sample_zero_when_equal():
    target = torch.randn(2, 3, 4)
    per_sample = reduce_vector_flow_per_sample(target, target)
    assert torch.allclose(per_sample, torch.zeros_like(per_sample))


def test_policy_default_peft_targets_vector_mode_include_latent_vector_modules():
    policy = LatentSmolVLAPolicy.__new__(LatentSmolVLAPolicy)
    policy.config = SimpleNamespace(latent_head_mode="vector_diffusion")

    targets = policy._get_default_peft_targets()

    assert "latent_in_proj" in targets["target_modules"]
    assert "latent_time_mlp_in" in targets["target_modules"]
    assert "latent_time_mlp_out" in targets["target_modules"]
    assert "latent_vector_out_proj" in targets["target_modules"]
    assert targets["modules_to_save"] == []


def test_policy_default_peft_targets_index_ce_include_discrete_modules():
    policy = LatentSmolVLAPolicy.__new__(LatentSmolVLAPolicy)
    policy.config = SimpleNamespace(latent_head_mode="index_cross_entropy")

    targets = policy._get_default_peft_targets()

    assert "latent_id_out_proj" in targets["target_modules"]
    assert targets["modules_to_save"] == ["model.latent_id_query_embed"]


def test_policy_default_peft_targets_vector_mse_include_query_modules():
    policy = LatentSmolVLAPolicy.__new__(LatentSmolVLAPolicy)
    policy.config = SimpleNamespace(latent_head_mode="vector_mse")

    targets = policy._get_default_peft_targets()

    assert "latent_vector_out_proj" in targets["target_modules"]
    assert targets["modules_to_save"] == ["model.latent_vector_query_embed"]


def test_model_forward_returns_unreduced_action_losses_shape():
    model = LatentSmolVLAFlowMatching.__new__(LatentSmolVLAFlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(chunk_size=4, max_action_dim=6)
    model.action_out_proj = nn.Linear(7, 6)
    model.sample_noise = lambda shape, device: torch.zeros(shape, device=device)
    model.sample_time = lambda batch_size, device: torch.full((batch_size,), 0.5, device=device)
    model.embed_prefix = lambda *args, **kwargs: (
        torch.zeros(2, 3, 7),
        torch.ones(2, 3, dtype=torch.bool),
        torch.zeros(2, 3, dtype=torch.bool),
    )
    model.embed_suffix = lambda noisy_actions, timestep: (
        torch.zeros(noisy_actions.shape[0], noisy_actions.shape[1], 7),
        torch.ones(noisy_actions.shape[0], noisy_actions.shape[1], dtype=torch.bool),
        torch.ones(noisy_actions.shape[0], noisy_actions.shape[1]),
    )

    def fake_forward(**kwargs):
        prefix_embs, suffix_embs = kwargs["inputs_embeds"]
        assert prefix_embs is not None
        assert suffix_embs is not None
        assert kwargs["fill_kv_cache"] is False
        return ([prefix_embs, torch.ones_like(suffix_embs)], None)

    model.vlm_with_expert = SimpleNamespace(forward=fake_forward)

    actions = torch.randn(2, 4, 6)
    losses = model.forward(None, None, None, None, None, actions)
    assert losses.shape == actions.shape


def test_forward_latent_id_logits_uses_suffix_queries():
    model = LatentSmolVLAFlowMatching.__new__(LatentSmolVLAFlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(latent_code_seq_len=3, latent_codebook_size=5)
    model.latent_id_out_proj = nn.Linear(7, 5)
    model.embed_prefix = lambda *args, **kwargs: (
        torch.zeros(2, 4, 7),
        torch.ones(2, 4, dtype=torch.bool),
        torch.zeros(2, 4, dtype=torch.bool),
    )
    model.embed_latent_id_suffix = lambda *, batch_size, device, dtype: (
        torch.zeros(batch_size, 3, 7, dtype=dtype, device=device),
        torch.ones(batch_size, 3, dtype=torch.bool, device=device),
        torch.ones(batch_size, 3, dtype=dtype, device=device),
    )

    def fake_forward(**kwargs):
        assert kwargs["inputs_embeds"][1] is not None
        assert kwargs["fill_kv_cache"] is False
        suffix_embs = kwargs["inputs_embeds"][1]
        return ([None, torch.ones_like(suffix_embs)], None)

    model.vlm_with_expert = SimpleNamespace(forward=fake_forward)

    logits = model.forward_latent_id_logits(None, None, None, None, None)
    assert logits.shape == (2, 3, 5)


def test_forward_latent_vector_losses_returns_unreduced_shape():
    model = LatentSmolVLAFlowMatching.__new__(LatentSmolVLAFlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        latent_code_seq_len=3,
        latent_vector_dim=12,
        latent_flow_beta_alpha=1.5,
        latent_flow_beta_beta=1.0,
    )
    model.latent_vector_out_proj = nn.Linear(7, 4)
    model.embed_prefix = lambda *args, **kwargs: (
        torch.zeros(2, 4, 7),
        torch.ones(2, 4, dtype=torch.bool),
        torch.zeros(2, 4, dtype=torch.bool),
    )
    model.embed_latent_vector_suffix = lambda noisy_latents, timestep: (
        torch.zeros(noisy_latents.shape[0], noisy_latents.shape[1], 7),
        torch.ones(noisy_latents.shape[0], noisy_latents.shape[1], dtype=torch.bool),
        torch.ones(noisy_latents.shape[0], noisy_latents.shape[1]),
    )

    def fake_forward(**kwargs):
        assert kwargs["inputs_embeds"][1] is not None
        suffix_embs = kwargs["inputs_embeds"][1]
        return ([None, torch.ones_like(suffix_embs)], None)

    model.vlm_with_expert = SimpleNamespace(forward=fake_forward)

    latent_vectors = torch.randn(2, 3, 4)
    losses = model.forward_latent_vector_losses(
        None,
        None,
        None,
        None,
        None,
        latent_vectors,
        noise=torch.zeros_like(latent_vectors),
        time=torch.full((2,), 0.5),
    )
    assert losses.shape == latent_vectors.shape


def test_forward_latent_vector_predictions_use_suffix_queries():
    model = LatentSmolVLAFlowMatching.__new__(LatentSmolVLAFlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(latent_code_seq_len=3)
    model.latent_vector_out_proj = nn.Linear(7, 4)
    model.embed_prefix = lambda *args, **kwargs: (
        torch.zeros(2, 4, 7),
        torch.ones(2, 4, dtype=torch.bool),
        torch.zeros(2, 4, dtype=torch.bool),
    )
    model.embed_latent_vector_query_suffix = lambda *, batch_size, device, dtype: (
        torch.zeros(batch_size, 3, 7, dtype=dtype, device=device),
        torch.ones(batch_size, 3, dtype=torch.bool, device=device),
        torch.ones(batch_size, 3, dtype=dtype, device=device),
    )

    def fake_forward(**kwargs):
        assert kwargs["inputs_embeds"][1] is not None
        assert kwargs["fill_kv_cache"] is False
        suffix_embs = kwargs["inputs_embeds"][1]
        return ([None, torch.ones_like(suffix_embs)], None)

    model.vlm_with_expert = SimpleNamespace(forward=fake_forward)

    predictions = model.forward_latent_vector_predictions(None, None, None, None, None)
    assert predictions.shape == (2, 3, 4)


def test_forward_latent_mse_branch_runs_and_reports_mode_metrics():
    policy = LatentSmolVLAPolicy.__new__(LatentSmolVLAPolicy)
    policy.config = SimpleNamespace(
        latent_head_mode="vector_mse",
        latent_label_key="latent_labels.continuous_vector_latents",
        latent_code_seq_len=3,
        latent_vector_dim=12,
        latent_valid_key=None,
        latent_supervision_key=None,
    )
    policy.prepare_images = lambda batch: ([torch.zeros(2, 3, 4, 4)], [torch.ones(2, dtype=torch.bool)])
    policy.prepare_state = lambda batch: torch.zeros(2, 6)
    policy.model = SimpleNamespace(
        forward_latent_vector_predictions=lambda *args, **kwargs: torch.zeros(2, 3, 4)
    )

    batch = {
        "latent_labels.continuous_vector_latents": torch.randn(2, 3, 4),
        "observation.language.tokens": torch.zeros(2, 5, dtype=torch.long),
        "observation.language.attention_mask": torch.ones(2, 5, dtype=torch.bool),
    }

    latent_loss, metrics = policy._forward_latent_mse_branch(batch)

    assert latent_loss.ndim == 0
    assert metrics["latent_head_mode_index_cross_entropy"] == 0.0
    assert metrics["latent_head_mode_vector_diffusion"] == 0.0
    assert metrics["latent_head_mode_vector_mse"] == 1.0

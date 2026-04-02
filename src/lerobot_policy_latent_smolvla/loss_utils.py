import torch
import torch.nn.functional as F  # noqa: N812


def pool_hidden(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean pooling over the sequence dimension."""
    mask_float = mask.float().unsqueeze(-1)
    return (hidden * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1e-9)


def make_sample_keep_mask(
    batch: dict,
    *,
    key: str | None,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if key is None:
        return torch.ones(batch_size, dtype=torch.bool, device=device)
    if key not in batch:
        raise KeyError(f"Missing supervision key {key!r} in batch")

    values = batch[key]
    if torch.is_tensor(values):
        mask = values.to(device=device, dtype=torch.bool)
    else:
        mask = torch.as_tensor(values, device=device, dtype=torch.bool)
    return mask.reshape(batch_size)


def reduce_action_per_sample(
    losses: torch.Tensor,
    *,
    max_action_dim: int,
    action_is_pad: torch.Tensor | None,
) -> torch.Tensor:
    if action_is_pad is not None:
        losses = losses * (~action_is_pad).unsqueeze(-1)
    losses = losses[:, :, :max_action_dim]
    return losses.mean(dim=(1, 2))


def reduce_latent_per_sample(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if labels.ndim == 1 and logits.ndim == 3 and logits.shape[1] == 1:
        labels = labels.unsqueeze(1)
    if labels.ndim != 2:
        raise ValueError(f"Expected latent labels with shape [B, S], got {tuple(labels.shape)}")
    if logits.ndim != 3:
        raise ValueError(f"Expected latent logits with shape [B, S, K], got {tuple(logits.shape)}")
    if logits.shape[:2] != labels.shape:
        raise ValueError(
            f"Latent logits/labels shape mismatch: logits={tuple(logits.shape)}, labels={tuple(labels.shape)}"
        )

    valid_tokens = labels.ne(ignore_index)
    flat_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).reshape_as(labels)
    flat_loss = flat_loss * valid_tokens

    valid_counts = valid_tokens.sum(dim=1)
    per_sample_loss = flat_loss.sum(dim=1) / valid_counts.clamp(min=1)

    preds = logits.argmax(dim=-1)
    correct = (preds == labels) & valid_tokens
    per_sample_acc = correct.sum(dim=1).float() / valid_counts.clamp(min=1).float()

    probs = F.softmax(logits, dim=-1)
    pred_probs = probs.gather(-1, preds.unsqueeze(-1)).squeeze(-1) * valid_tokens
    per_sample_conf = pred_probs.sum(dim=1) / valid_counts.clamp(min=1).float()

    valid_samples = valid_counts.gt(0)
    return per_sample_loss, valid_samples, per_sample_acc, per_sample_conf


def reshape_latent_vector_sequence(
    vectors: torch.Tensor,
    *,
    latent_code_seq_len: int,
    latent_vector_dim: int,
) -> torch.Tensor:
    if vectors.ndim == 2:
        if int(vectors.shape[1]) != int(latent_vector_dim):
            raise ValueError(
                "Expected latent vector tensor [B, latent_vector_dim], "
                f"got shape {tuple(vectors.shape)} and latent_vector_dim={latent_vector_dim}"
            )
        latent_step_dim = int(latent_vector_dim) // int(latent_code_seq_len)
        return vectors.reshape(vectors.shape[0], int(latent_code_seq_len), latent_step_dim)
    if vectors.ndim == 3:
        latent_step_dim = int(latent_vector_dim) // int(latent_code_seq_len)
        expected = (int(latent_code_seq_len), latent_step_dim)
        got = (int(vectors.shape[1]), int(vectors.shape[2]))
        if got != expected:
            raise ValueError(
                f"Expected latent vector tensor [B,{expected[0]},{expected[1]}], got {tuple(vectors.shape)}"
            )
        return vectors
    raise ValueError(f"Expected latent vectors rank 2 or 3, got rank={vectors.ndim}")


def sample_beta_time(
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    beta_dist = torch.distributions.Beta(concentration1=alpha, concentration0=beta)
    time_beta = beta_dist.sample((batch_size,)).to(device=device, dtype=dtype)
    return time_beta * 0.999 + 0.001


def reduce_vector_flow_per_sample(
    predicted_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
) -> torch.Tensor:
    if predicted_velocity.shape != target_velocity.shape:
        raise ValueError(
            "Vector flow tensors must have identical shape, "
            f"got predicted={tuple(predicted_velocity.shape)} target={tuple(target_velocity.shape)}"
        )
    losses = F.mse_loss(predicted_velocity, target_velocity, reduction="none")
    return losses.mean(dim=tuple(range(1, losses.ndim)))


def masked_mean_or_zero(values: torch.Tensor, keep: torch.Tensor) -> tuple[torch.Tensor, int]:
    kept = int(keep.sum().item())
    if kept > 0:
        return values[keep].mean(), kept
    return values.sum() * 0.0, 0

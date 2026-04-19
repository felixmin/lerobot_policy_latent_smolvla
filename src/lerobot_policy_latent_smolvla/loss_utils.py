import torch


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


def make_sequence_keep_mask(
    batch: dict,
    *,
    key: str | None,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> torch.Tensor:
    if key is None:
        return torch.ones(batch_size, sequence_length, dtype=torch.bool, device=device)
    if key not in batch:
        raise KeyError(f"Missing supervision key {key!r} in batch")

    values = batch[key]
    if torch.is_tensor(values):
        mask = values.to(device=device, dtype=torch.bool)
    else:
        mask = torch.as_tensor(values, device=device, dtype=torch.bool)

    if mask.ndim == 0:
        raise ValueError(f"Expected sequence keep mask rank >= 1, got scalar for key {key!r}")
    if int(mask.shape[0]) != int(batch_size):
        raise ValueError(
            f"Expected sequence keep mask batch size {batch_size}, got shape {tuple(mask.shape)}"
        )
    if mask.ndim == 1:
        return mask.reshape(batch_size, 1).expand(batch_size, sequence_length)
    if mask.ndim == 2:
        if int(mask.shape[1]) == 1:
            return mask.expand(batch_size, sequence_length)
        if int(mask.shape[1]) == int(sequence_length):
            return mask
    if mask.ndim == 3 and int(mask.shape[1]) == int(sequence_length) and int(mask.shape[2]) == 1:
        return mask.squeeze(-1)
    raise ValueError(
        "Expected sequence keep mask shaped [B], [B,1], [B,S], or [B,S,1], "
        f"got {tuple(mask.shape)} for key {key!r} and S={sequence_length}"
    )


def expand_keep_mask(values: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    if int(values.shape[0]) != int(keep.shape[0]):
        raise ValueError(
            f"Expected values and keep mask to share batch dimension, got {values.shape[0]} and {keep.shape[0]}"
        )
    keep = keep.bool()
    if keep.ndim == 1:
        return keep.reshape(values.shape[0], *([1] * (values.ndim - 1))).expand_as(values)
    if keep.ndim > values.ndim:
        raise ValueError(
            f"Keep mask rank must not exceed values rank, got keep={keep.ndim} values={values.ndim}"
        )
    if values.shape[: keep.ndim] != keep.shape:
        raise ValueError(
            f"Keep mask shape {tuple(keep.shape)} is incompatible with values shape {tuple(values.shape)}"
        )
    view_shape = tuple(keep.shape) + (1,) * (values.ndim - keep.ndim)
    return keep.reshape(view_shape).expand_as(values)


def reduce_action_per_sample(
    losses: torch.Tensor,
    *,
    feature_dim: int,
    action_is_pad: torch.Tensor | None,
) -> torch.Tensor:
    if action_is_pad is not None:
        losses = losses * (~action_is_pad).unsqueeze(-1)
    losses = losses[:, :, :feature_dim]
    return losses.mean(dim=(1, 2))


def reshape_latent_vector_sequence(
    vectors: torch.Tensor,
    *,
    latent_sequence_length: int | None = None,
) -> torch.Tensor:
    if vectors.ndim == 2:
        if latent_sequence_length is None:
            raise ValueError(
                "Flat latent vectors require latent_sequence_length to reshape [B, D] into [B, S, D_token]."
            )
        if int(vectors.shape[1]) % int(latent_sequence_length) != 0:
            raise ValueError(
                "Expected flat latent vector tensor [B, S*D_token] with total width divisible by latent_sequence_length, "
                f"got shape {tuple(vectors.shape)} and latent_sequence_length={latent_sequence_length}"
            )
        latent_step_dim = int(vectors.shape[1]) // int(latent_sequence_length)
        return vectors.reshape(vectors.shape[0], int(latent_sequence_length), latent_step_dim)
    if vectors.ndim >= 3:
        if latent_sequence_length is not None and int(vectors.shape[1]) != int(latent_sequence_length):
            raise ValueError(
                f"Expected latent vector tensor with sequence length {latent_sequence_length}, got shape {tuple(vectors.shape)}"
            )
        batch_size = int(vectors.shape[0])
        sequence_length = int(vectors.shape[1])
        latent_step_dim = 1
        for dim in vectors.shape[2:]:
            latent_step_dim *= int(dim)
        return vectors.reshape(batch_size, sequence_length, latent_step_dim)
    raise ValueError(f"Expected latent vectors rank >= 2, got rank={vectors.ndim}")


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
def masked_mean_or_zero(values: torch.Tensor, keep: torch.Tensor) -> tuple[torch.Tensor, int]:
    kept = int(keep.bool().sum().item())
    if kept > 0:
        expanded_keep = expand_keep_mask(values, keep)
        return values.masked_select(expanded_keep).mean(), kept
    return values.sum() * 0.0, 0

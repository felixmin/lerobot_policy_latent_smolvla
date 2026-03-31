"""Standalone latent SmolVLA LeRobot policy plugin."""

try:
    import lerobot  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "lerobot is not installed. Please install lerobot to use "
        "lerobot_policy_latent_smolvla."
    ) from exc

from lerobot_policy_latent_smolvla.configuration_latent_smolvla import LatentSmolVLAConfig
from lerobot_policy_latent_smolvla import processor_latent_smolvla as _processor_latent_smolvla  # noqa: F401

__all__ = [
    "LatentSmolVLAConfig",
    "LatentSmolVLAPolicy",
    "LatentSmolVLANewLineProcessor",
    "make_latent_smolvla_pre_post_processors",
]


def __getattr__(name: str):
    if name == "LatentSmolVLAPolicy":
        from lerobot_policy_latent_smolvla.modeling_latent_smolvla import LatentSmolVLAPolicy

        return LatentSmolVLAPolicy
    if name in {"LatentSmolVLANewLineProcessor", "make_latent_smolvla_pre_post_processors"}:
        from lerobot_policy_latent_smolvla.processor_latent_smolvla import (
            LatentSmolVLANewLineProcessor,
            make_latent_smolvla_pre_post_processors,
        )

        mapping = {
            "LatentSmolVLANewLineProcessor": LatentSmolVLANewLineProcessor,
            "make_latent_smolvla_pre_post_processors": make_latent_smolvla_pre_post_processors,
        }
        return mapping[name]
    raise AttributeError(name)

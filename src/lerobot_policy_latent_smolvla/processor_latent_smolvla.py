#!/usr/bin/env python

from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    ComplementaryDataProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import batch_to_transition, policy_action_to_transition, transition_to_policy_action
from lerobot.types import TransitionKey
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from lerobot_policy_latent_smolvla.configuration_latent_smolvla import LatentSmolVLAConfig


def make_latent_smolvla_pre_post_processors(
    config: LatentSmolVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        LatentSmolVLANewLineProcessor(),
        TokenizerProcessorStep(
            tokenizer_name=config.vlm_model_name,
            padding=config.pad_language_to,
            padding_side="right",
            max_length=config.tokenizer_max_length,
        ),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        LatentSmolVLALatentTargetNormalizer(
            latent_label_key=config.latent_label_key,
            enabled=bool(config.normalize_latent_targets),
            eps=float(config.latent_normalization_eps),
            stats=None if dataset_stats is None else dataset_stats.get(config.latent_label_key),
        ),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
            to_transition=_make_batch_to_transition_with_latent_keys(config),
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


def _make_batch_to_transition_with_latent_keys(
    config: LatentSmolVLAConfig,
):
    preserved_keys = {
        key
        for key in [
            config.latent_label_key,
            config.latent_valid_key,
            config.latent_supervision_key,
            config.action_supervision_key,
        ]
        if key
    }

    def _to_transition(batch: dict[str, Any]):
        transition = batch_to_transition(batch)
        complementary_data = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA, {}))
        for key in preserved_keys:
            if key in batch:
                complementary_data[key] = batch[key]
        transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
        return transition

    return _to_transition


@ProcessorStepRegistry.register(name="latent_smolvla_new_line_processor")
class LatentSmolVLANewLineProcessor(ComplementaryDataProcessorStep):
    """Ensure the task string keeps the newline convention used by SmolVLA."""

    def complementary_data(self, complementary_data):
        if "task" not in complementary_data:
            return complementary_data

        task = complementary_data["task"]
        if task is None:
            return complementary_data

        new_complementary_data = dict(complementary_data)
        if isinstance(task, str):
            if not task.endswith("\n"):
                new_complementary_data["task"] = f"{task}\n"
        elif isinstance(task, list) and all(isinstance(t, str) for t in task):
            new_complementary_data["task"] = [
                t if t.endswith("\n") else f"{t}\n" for t in task
            ]
        return new_complementary_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register(name="latent_smolvla_latent_target_normalizer")
class LatentSmolVLALatentTargetNormalizer(ComplementaryDataProcessorStep):
    latent_label_key: str
    enabled: bool = True
    eps: float = 1e-8
    stats: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None
        if self.stats is not None:
            self._load_stats(self.stats)

    def _load_stats(self, stats: dict[str, Any]) -> None:
        mean = stats.get("mean")
        std = stats.get("std")
        if mean is None or std is None:
            return
        self._mean = torch.as_tensor(mean)
        self._std = torch.as_tensor(std)

    def _normalize_latent_labels(self, latent_labels: torch.Tensor) -> torch.Tensor:
        if self._mean is None or self._std is None:
            raise ValueError(
                f"Latent normalization is enabled, but stats are missing for {self.latent_label_key!r}."
            )

        mean = self._mean.to(device=latent_labels.device, dtype=latent_labels.dtype)
        std = self._std.to(device=latent_labels.device, dtype=latent_labels.dtype)

        if latent_labels.shape[-mean.ndim :] == mean.shape:
            return (latent_labels - mean) / (std + float(self.eps))

        flat_feature_dim = 1
        for dim in mean.shape:
            flat_feature_dim *= int(dim)

        if latent_labels.ndim >= 1 and int(latent_labels.shape[-1]) == flat_feature_dim:
            normalized = (
                latent_labels.reshape(*latent_labels.shape[:-1], *mean.shape) - mean
            ) / (std + float(self.eps))
            return normalized.reshape_as(latent_labels)

        trailing_numel = 1
        for start_idx in range(latent_labels.ndim - 1, -1, -1):
            trailing_numel *= int(latent_labels.shape[start_idx])
            if trailing_numel == flat_feature_dim:
                labels_flat = latent_labels.reshape(*latent_labels.shape[:start_idx], flat_feature_dim)
                mean_flat = mean.reshape(flat_feature_dim)
                std_flat = std.reshape(flat_feature_dim)
                normalized = (labels_flat - mean_flat) / (std_flat + float(self.eps))
                return normalized.reshape_as(latent_labels)
            if trailing_numel > flat_feature_dim:
                break

        raise ValueError(
            "Latent normalization stats shape is incompatible with latent labels: "
            f"labels={tuple(latent_labels.shape)} stats={tuple(mean.shape)}"
        )

    def complementary_data(self, complementary_data):
        if not self.enabled:
            return complementary_data
        if self.latent_label_key not in complementary_data:
            return complementary_data

        latent_labels = complementary_data[self.latent_label_key]
        if not torch.is_tensor(latent_labels):
            latent_labels = torch.as_tensor(latent_labels)
        if not torch.is_floating_point(latent_labels):
            return complementary_data

        new_complementary_data = dict(complementary_data)
        new_complementary_data[self.latent_label_key] = self._normalize_latent_labels(latent_labels)
        return new_complementary_data

    def get_config(self) -> dict[str, Any]:
        return {
            "latent_label_key": self.latent_label_key,
            "enabled": self.enabled,
            "eps": self.eps,
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        if self._mean is None or self._std is None:
            return {}
        return {"mean": self._mean, "std": self._std}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self._mean = state.get("mean")
        self._std = state.get("std")

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

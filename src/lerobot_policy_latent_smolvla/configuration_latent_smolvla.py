from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.constants import OBS_IMAGES


@PreTrainedConfig.register_subclass("latent_smolvla")
@dataclass
class LatentSmolVLAConfig(PreTrainedConfig):
    """Standalone SmolVLA variant with hierarchical latent-plan and action diffusion."""

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Shorter state, action, and latent vectors will be padded.
    max_state_dim: int = 32
    max_action_dim: int = 32
    max_latent_dim: int | None = None

    # Image preprocessing.
    resize_imgs_with_padding: tuple[int, int] = (512, 512)
    empty_cameras: int = 0

    # Aloha compatibility.
    adapt_to_pi_aloha: bool = False
    use_delta_joint_actions_aloha: bool = False

    # Tokenizer / decoding.
    tokenizer_max_length: int = 48
    num_steps: int = 10
    use_cache: bool = True

    # Finetuning settings.
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    train_state_proj: bool = True

    # Training presets.
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 100_000
    scheduler_decay_lr: float = 2.5e-6

    # Backbone.
    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    load_vlm_weights: bool = True
    add_image_special_tokens: bool = False
    attention_mode: str = "cross_attn"
    prefix_length: int = -1
    pad_language_to: str = "longest"
    num_expert_layers: int = -1
    num_vlm_layers: int = 16
    self_attn_every_n_layers: int = 2
    expert_width_multiplier: float = 0.75
    min_period: float = 4e-3
    max_period: float = 4.0
    compile_model: bool = False
    compile_mode: str = "max-autotune"

    # Real-Time Chunking.
    rtc_config: RTCConfig | None = None

    # Latent supervision.
    training_mode: str = "multitask"
    latent_head_mode: str = "vector_diffusion"
    action_loss_weight: float = 1.0
    latent_loss_weight: float = 1.0

    latent_teacher_force_ratio_start: float = 1.0
    latent_teacher_force_ratio_end: float = 0.0
    latent_teacher_force_decay_steps: int = 100_000
    # Keep latent-related batch keys outside observation.* and route them explicitly
    # through latent_delta_indices so they can use future timestamp access without
    # sharing observation_delta_indices.
    latent_delta_indices: list[int] | None = None
    latent_label_key: str = "latent_labels.continuous_vector_latents"
    latent_valid_key: str | None = "latent_labels.valid"
    latent_supervision_key: str | None = None
    action_supervision_key: str | None = None
    normalize_latent_targets: bool = True
    latent_normalization_source: str = "latent"
    latent_normalization_eps: float = 1e-8
    latent_flow_beta_alpha: float = 1.5
    latent_flow_beta_beta: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.max_latent_dim is None:
            self.max_latent_dim = int(self.max_action_dim)

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError(
                "`use_delta_joint_actions_aloha` is used by smolvla for aloha real models. "
                "It is not ported yet in LeRobot."
            )

        if self.training_mode not in {"action", "latent", "multitask"}:
            raise ValueError(
                "training_mode must be one of {'action', 'latent', 'multitask'}, "
                f"got {self.training_mode!r}"
            )
        if self.latent_head_mode != "vector_diffusion":
            raise ValueError(
                "latent_head_mode must be 'vector_diffusion', "
                f"got {self.latent_head_mode!r}"
            )
        if self.latent_normalization_source not in {"latent", "action"}:
            raise ValueError(
                "latent_normalization_source must be one of {'latent', 'action'}, "
                f"got {self.latent_normalization_source!r}"
            )
        if self.action_loss_weight < 0.0:
            raise ValueError(
                f"action_loss_weight must be >= 0, got {self.action_loss_weight}"
            )
        if self.latent_loss_weight < 0.0:
            raise ValueError(
                f"latent_loss_weight must be >= 0, got {self.latent_loss_weight}"
            )
        if self.max_latent_dim < 1:
            raise ValueError(
                f"max_latent_dim must be >= 1, got {self.max_latent_dim}"
            )
        if (
            self.latent_delta_indices is not None
            and len(self.latent_delta_indices) < 1
        ):
            raise ValueError(
                "latent_delta_indices must be non-empty when provided, "
                f"got {self.latent_delta_indices}"
            )
        if self.latent_flow_beta_alpha <= 0.0 or self.latent_flow_beta_beta <= 0.0:
            raise ValueError(
                "latent_flow_beta_alpha and latent_flow_beta_beta must both be > 0"
            )
        if not 0.0 <= self.latent_teacher_force_ratio_start <= 1.0:
            raise ValueError(
                "latent_teacher_force_ratio_start must be in [0, 1], "
                f"got {self.latent_teacher_force_ratio_start}"
            )
        if not 0.0 <= self.latent_teacher_force_ratio_end <= 1.0:
            raise ValueError(
                "latent_teacher_force_ratio_end must be in [0, 1], "
                f"got {self.latent_teacher_force_ratio_end}"
            )
        if self.latent_teacher_force_decay_steps < 0:
            raise ValueError(
                "latent_teacher_force_decay_steps must be >= 0, "
                f"got {self.latent_teacher_force_decay_steps}"
            )
        if self.training_mode in {"action", "multitask"} and self.action_loss_weight == 0.0:
            raise ValueError(
                "action_loss_weight must be > 0 when training_mode uses action supervision"
            )
        if self.training_mode in {"latent", "multitask"} and self.latent_loss_weight == 0.0:
            raise ValueError(
                "latent_loss_weight must be > 0 when training_mode uses latent supervision"
            )

    def validate_features(self) -> None:
        if self.input_features is None:
            self.input_features = {}
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            self.input_features[key] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list:
        return [0]

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    @property
    def latent_sequence_length(self) -> int:
        if self.latent_delta_indices is not None:
            return len(self.latent_delta_indices)
        return self.chunk_size

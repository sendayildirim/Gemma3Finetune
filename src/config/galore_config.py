"""
Configurations for GaLore fine-tuning
Memory efficient training with gradient low-rank projection
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GaLoreConfig:
    # Model settings
    model_name: str = "google/gemma-3-1b-it"
    max_seq_length: int = 2048

    # GaLore parameters
    rank: int = 128
    update_proj_gap: int = 200
    galore_scale: float = 0.25
    proj_type: str = "std"

    # Training parameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100

    # Optimizer settings
    optim: str = "galore_adamw"
    lr_scheduler_type: str = "cosine"

    # Logging and checkpoint
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    output_dir: str = "./models/galore"

    # Performance settings
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0

    # Seed
    seed: int = 42

    # GaLore specific layer targeting
    target_modules_list: list = None

    def __post_init__(self):
        if self.target_modules_list is None:
            # Layers for GaLore application in Gemma model
            self.target_modules_list = ["attn", "mlp"]


def get_galore_config() -> GaLoreConfig:
    """Returns GaLore configuration"""
    return GaLoreConfig()

"""
Configurations for QLoRA fine-tuning
Memory efficient training with 4-bit quantization
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class QLoRAConfig:
    # Model settings
    model_name: str = "google/gemma-3-1b-it"
    max_seq_length: int = 2048

    # LoRA parameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = None

    # 4-bit quantization parameters
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # Training parameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100

    # Optimizer settings
    optim: str = "paged_adamw_8bit"
    lr_scheduler_type: str = "cosine"

    # Logging and checkpoint
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    output_dir: str = "./models/qlora"

    # Performance settings
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    max_grad_norm: float = 0.3

    # Seed
    seed: int = 42

    def __post_init__(self):
        if self.target_modules is None:
            # Target modules for Gemma model
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def get_qlora_config() -> QLoRAConfig:
    """Returns QLoRA configuration"""
    return QLoRAConfig()

"""
GaLore fine-tuning script
Memory efficient training with gradient low-rank projection
"""

import os
import sys
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from trl import SFTTrainer

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config.galore_config import get_galore_config
from src.training.utils import (
    MemoryTracker,
    TrainingTimer,
    save_training_metrics,
    print_trainable_parameters,
    setup_wandb_env,
    get_device_info,
    print_gpu_utilization
)


def load_model_and_tokenizer(config):
    """
    Loads model and tokenizer (full precision)

    Args:
        config: GaLore configuration

    Returns:
        (model, tokenizer) tuple
    """
    print("Loading model and tokenizer")
    print(f"Model: {config.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )

    print("Model loaded")
    print_trainable_parameters(model)

    return model, tokenizer


def load_processed_dataset(dataset_path: str):
    """
    Loads preprocessed dataset

    Args:
        dataset_path: Dataset JSON path

    Returns:
        Dataset
    """
    print(f"\nLoading dataset: {dataset_path}")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    print(f"Dataset size: {len(dataset)}")

    return dataset


def create_galore_optimizer(model, config):
    """
    Creates GaLore optimizer

    Args:
        model: Model
        config: GaLore configuration

    Returns:
        Optimizer
    """
    try:
        from galore_torch import GaLoreAdamW

        print("\nCreating GaLore optimizer")

        decay_params = []
        nodecay_params = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                if "bias" in name or "norm" in name or "ln" in name:
                    nodecay_params.append(param)
                else:
                    decay_params.append(param)

        optimizer = GaLoreAdamW(
            [
                {
                    "params": decay_params,
                    "weight_decay": config.weight_decay,
                    "rank": config.rank,
                    "update_proj_gap": config.update_proj_gap,
                    "scale": config.galore_scale,
                    "proj_type": config.proj_type,
                },
                {
                    "params": nodecay_params,
                    "weight_decay": 0.0,
                },
            ],
            lr=config.learning_rate,
        )

        print("GaLore optimizer created")
        return optimizer

    except ImportError:
        print("WARNING: galore_torch library not found")
        print("Will use normal AdamW optimizer")
        return None


def train(config=None):
    """
    GaLore fine-tuning main function

    Args:
        config: GaLore configuration
    """
    if config is None:
        config = get_galore_config()

    print("="*60)
    print("GaLore Fine-Tuning Starting")
    print("="*60)

    setup_wandb_env()

    print("\nDevice information:")
    device_info = get_device_info()
    for key, value in device_info.items():
        print(f"  {key}: {value}")

    print("\nGPU usage:")
    print_gpu_utilization()

    memory_tracker = MemoryTracker()
    timer = TrainingTimer()

    timer.start()
    memory_tracker.reset()

    model, tokenizer = load_model_and_tokenizer(config)

    memory_tracker.update()
    print(f"\nGPU memory after model loading: {memory_tracker.get_peak_memory_gb():.2f} GB")

    train_dataset_path = os.path.join(
        config.output_dir.replace("models/galore", "data/processed"),
        "train_processed.json"
    )
    train_dataset = load_processed_dataset(train_dataset_path)

    print("\nSetting up training arguments")
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        bf16=config.bf16,
        fp16=config.fp16,
        lr_scheduler_type=config.lr_scheduler_type,
        gradient_checkpointing=config.gradient_checkpointing,
        max_grad_norm=config.max_grad_norm,
        report_to="none",
        save_total_limit=2,
        seed=config.seed,
    )

    galore_optimizer = create_galore_optimizer(model, config)

    print("\nCreating SFTTrainer")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
        optimizers=(galore_optimizer, None) if galore_optimizer else (None, None),
    )

    print("\n" + "="*60)
    print("Training Starting")
    print("="*60)

    trainer.train()

    timer.stop()
    memory_tracker.update()

    print("\n" + "="*60)
    print("Training Completed")
    print("="*60)

    print("\nSaving model")
    trainer.save_model(os.path.join(config.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(config.output_dir, "final"))

    memory_stats = memory_tracker.get_memory_stats()
    training_time_hours = timer.get_elapsed_time_hours()

    print("\n" + "="*60)
    print("Training Metrics")
    print("="*60)
    print(f"Training duration: {timer.get_elapsed_time_formatted()}")
    print(f"Peak GPU memory: {memory_stats['peak_memory_allocated_gb']:.2f} GB")

    metrics = {
        "technique": "GaLore",
        "model": config.model_name,
        "training_time_hours": training_time_hours,
        "memory_stats": memory_stats,
        "config": {
            "rank": config.rank,
            "update_proj_gap": config.update_proj_gap,
            "galore_scale": config.galore_scale,
            "learning_rate": config.learning_rate,
            "num_train_epochs": config.num_train_epochs,
            "batch_size": config.per_device_train_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
        }
    }

    metrics_dir = config.output_dir.replace("models/galore", "results/metrics")
    save_training_metrics(metrics_dir, "galore", metrics)

    print("\nGaLore fine-tuning completed")


if __name__ == "__main__":
    train()

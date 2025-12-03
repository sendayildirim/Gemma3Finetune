"""
QLoRA fine-tuning script
Memory efficient LoRA training with 4-bit quantization
"""

import os
import sys
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config.qlora_config import get_qlora_config
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
    Loads model and tokenizer (with 4-bit quantization)

    Args:
        config: QLoRA configuration

    Returns:
        (model, tokenizer) tuple
    """
    print("Loading model and tokenizer")
    print(f"Model: {config.model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    print("Model loaded")
    return model, tokenizer


def setup_lora(model, config):
    """
    Adds LoRA adapters to model

    Args:
        model: Base model
        config: QLoRA configuration

    Returns:
        Model configured with LoRA
    """
    print("\nSetting up LoRA configuration")

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    print("LoRA configuration completed")
    print_trainable_parameters(model)

    return model


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


def train(config=None):
    """
    QLoRA fine-tuning main function

    Args:
        config: QLoRA configuration
    """
    if config is None:
        config = get_qlora_config()

    print("="*60)
    print("QLoRA Fine-Tuning Starting")
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

    model = setup_lora(model, config)

    memory_tracker.update()

    train_dataset_path = os.path.join(
        config.output_dir.replace("models/qlora", "data/processed"),
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
        optim=config.optim,
        lr_scheduler_type=config.lr_scheduler_type,
        gradient_checkpointing=config.gradient_checkpointing,
        max_grad_norm=config.max_grad_norm,
        report_to="none",
        save_total_limit=2,
        seed=config.seed,
    )

    print("\nCreating SFTTrainer")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
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
        "technique": "QLoRA",
        "model": config.model_name,
        "training_time_hours": training_time_hours,
        "memory_stats": memory_stats,
        "config": {
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "learning_rate": config.learning_rate,
            "num_train_epochs": config.num_train_epochs,
            "batch_size": config.per_device_train_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
        }
    }

    metrics_dir = config.output_dir.replace("models/qlora", "results/metrics")
    save_training_metrics(metrics_dir, "qlora", metrics)

    print("\nQLoRA fine-tuning completed")


if __name__ == "__main__":
    train()

"""
Common utility functions for training
"""

import os
import json
import torch
import psutil
import GPUtil
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class MemoryTracker:
    """GPU and CPU memory tracking class"""

    def __init__(self):
        self.peak_memory_allocated = 0
        self.peak_memory_reserved = 0

    def update(self):
        """Updates memory usage"""
        if torch.cuda.is_available():
            self.peak_memory_allocated = max(
                self.peak_memory_allocated,
                torch.cuda.memory_allocated() / 1024**3
            )
            self.peak_memory_reserved = max(
                self.peak_memory_reserved,
                torch.cuda.memory_reserved() / 1024**3
            )

    def get_peak_memory_gb(self) -> float:
        """Returns peak GPU memory usage in GB"""
        return self.peak_memory_allocated

    def get_memory_stats(self) -> Dict[str, float]:
        """Returns detailed memory statistics"""
        stats = {
            "peak_memory_allocated_gb": self.peak_memory_allocated,
            "peak_memory_reserved_gb": self.peak_memory_reserved,
        }

        if torch.cuda.is_available():
            stats["current_memory_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
            stats["current_memory_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3

        return stats

    def reset(self):
        """Resets memory tracking"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.peak_memory_allocated = 0
        self.peak_memory_reserved = 0


class TrainingTimer:
    """Training time tracking class"""

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        """Starts the timer"""
        self.start_time = datetime.now()

    def stop(self):
        """Stops the timer"""
        self.end_time = datetime.now()

    def get_elapsed_time_hours(self) -> float:
        """Returns elapsed time in hours"""
        if self.start_time is None:
            return 0.0

        end = self.end_time if self.end_time else datetime.now()
        elapsed = end - self.start_time
        return elapsed.total_seconds() / 3600

    def get_elapsed_time_formatted(self) -> str:
        """Returns elapsed time as formatted string"""
        hours = self.get_elapsed_time_hours()
        h = int(hours)
        m = int((hours - h) * 60)
        s = int(((hours - h) * 60 - m) * 60)
        return f"{h}h {m}m {s}s"


def save_training_metrics(
    output_dir: str,
    technique: str,
    metrics: Dict[str, Any]
):
    """
    Saves training metrics

    Args:
        output_dir: Save directory
        technique: Fine-tuning technique name
        metrics: Metric dictionary
    """
    os.makedirs(output_dir, exist_ok=True)

    metrics_file = os.path.join(output_dir, f"{technique}_metrics.json")

    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved: {metrics_file}")


def print_trainable_parameters(model):
    """
    Prints trainable parameters in model

    Args:
        model: Pytorch model
    """
    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(
        f"Trainable params: {trainable_params:,} || "
        f"All params: {all_param:,} || "
        f"Trainable%: {100 * trainable_params / all_param:.2f}%"
    )


def setup_wandb_env():
    """Disables Wandb (not needed for assessment)"""
    os.environ["WANDB_DISABLED"] = "true"


def get_device_info() -> Dict[str, Any]:
    """
    Returns device information

    Returns:
        Device info dictionary
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda

    return info


def print_gpu_utilization():
    """Prints GPU utilization information"""
    if torch.cuda.is_available():
        GPUs = GPUtil.getGPUs()
        for gpu in GPUs:
            print(f"GPU {gpu.id}: {gpu.name}")
            print(f"  Memory Usage: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB ({gpu.memoryUtil*100:.1f}%)")
            print(f"  GPU Load: {gpu.load*100:.1f}%")
    else:
        print("No GPU available")

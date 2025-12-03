"""
Configurations for dataset and preprocessing
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class DataConfig:
    # Model
    model_name: str = "google/gemma-3-1b-it"

    # Dataset names
    alpaca_dataset: str = "tatsu-lab/alpaca"
    tulu_dataset: str = "allenai/tulu-v2-sft-mixture"
    ultrachat_dataset: str = "HuggingFaceH4/ultrachat_200k"

    # Sampling settings (over-sampling for preprocessing loss buffer)
    train_samples_per_dataset: int = 5200  # 4% buffer for filtering
    test_samples_per_dataset: int = 2100   # 5% buffer for filtering

    # Target sample counts (after balancing)
    target_train_per_dataset: int = 5000
    target_test_per_dataset: int = 2000

    # Total sample counts
    total_train_samples: int = 15000
    total_test_samples: int = 6000

    # Random seed
    seed: int = 42

    # Data paths
    raw_data_dir: str = "./data/raw"
    processed_data_dir: str = "./data/processed"
    samples_dir: str = "./data/samples"

    # Preprocessing settings
    max_length: int = 2048


def get_data_config() -> DataConfig:
    """Returns data configuration"""
    return DataConfig()

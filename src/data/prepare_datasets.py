"""
Dataset downloading and sampling process
Takes specified number of train and test samples from each dataset
Creates clean dataset with overlap control
"""

import os
import random
from typing import Dict, List, Tuple
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config.data_config import get_data_config


class DatasetPreparer:
    """Class for loading and sampling datasets"""

    def __init__(self, config=None):
        self.config = config if config else get_data_config()
        self.random_seed = self.config.seed
        random.seed(self.random_seed)

    def load_raw_datasets(self) -> Dict[str, Dataset]:
        """
        Loads all datasets from HuggingFace
        Returns: Dataset dictionary
        """
        print("Loading datasets")

        datasets = {}

        # Alpaca dataset
        print("Loading Alpaca")
        alpaca = load_dataset(self.config.alpaca_dataset, split="train")
        datasets["alpaca"] = alpaca
        print(f"Alpaca loaded: {len(alpaca)} samples")

        # Tulu v2 SFT dataset
        print("Loading Tulu v2")
        tulu = load_dataset(self.config.tulu_dataset, split="train")
        datasets["tulu"] = tulu
        print(f"Tulu v2 loaded: {len(tulu)} samples")

        # Ultrachat 200k dataset
        print("Loading Ultrachat 200k")
        ultrachat = load_dataset(self.config.ultrachat_dataset, split="train_sft")
        datasets["ultrachat"] = ultrachat
        print(f"Ultrachat loaded: {len(ultrachat)} samples")

        return datasets

    def sample_dataset(
        self,
        dataset: Dataset,
        n_train: int,
        n_test: int
    ) -> Tuple[Dataset, Dataset]:
        """
        Takes random samples from dataset and splits into train/test
        Guarantees no overlap

        Args:
            dataset: Source dataset
            n_train: Number of train samples
            n_test: Number of test samples

        Returns:
            (train_dataset, test_dataset) tuple
        """
        total_needed = n_train + n_test
        dataset_size = len(dataset)

        if total_needed > dataset_size:
            raise ValueError(
                f"Dataset size ({dataset_size}) not sufficient. "
                f"Required: {total_needed} (train: {n_train}, test: {n_test})"
            )

        # Select random indices
        all_indices = list(range(dataset_size))
        random.shuffle(all_indices)

        # Separate indices for train and test
        train_indices = all_indices[:n_train]
        test_indices = all_indices[n_train:n_train + n_test]

        # Overlap check
        assert len(set(train_indices) & set(test_indices)) == 0, "Train/test overlap exists"

        train_data = dataset.select(train_indices)
        test_data = dataset.select(test_indices)

        return train_data, test_data

    def format_alpaca_sample(self, sample: Dict) -> Dict:
        """
        Converts Alpaca format to common messages format
        """
        instruction = sample.get("instruction", "").strip()
        input_text = sample.get("input", "").strip()
        output_text = sample.get("output", "").strip()

        # Add input to instruction if exists
        if input_text:
            user_content = f"{instruction}\n\nInput: {input_text}"
        else:
            user_content = instruction

        return {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": output_text}
            ],
            "source_dataset": sample.get("source_dataset", "alpaca")
        }

    def format_tulu_sample(self, sample: Dict) -> Dict:
        """
        Converts Tulu v2 format to common messages format
        """
        messages = sample.get("messages", [])

        # Normalize messages
        normalized_messages = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "").strip()

            if role in ["user", "assistant"] and content:
                normalized_messages.append({
                    "role": role,
                    "content": content
                })

        return {
            "messages": normalized_messages,
            "source_dataset": sample.get("source_dataset", "tulu")
        }

    def format_ultrachat_sample(self, sample: Dict) -> Dict:
        """
        Converts Ultrachat format to common messages format
        """
        messages = sample.get("messages", [])

        # Normalize messages
        normalized_messages = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "").strip()

            if role in ["user", "assistant"] and content:
                normalized_messages.append({
                    "role": role,
                    "content": content
                })

        return {
            "messages": normalized_messages,
            "source_dataset": sample.get("source_dataset", "ultrachat")
        }

    def prepare_all_datasets(self) -> Tuple[Dataset, Dataset]:
        """
        Loads all datasets, samples and combines them

        Returns:
            (combined_train, combined_test) tuple
        """
        # Load raw datasets
        raw_datasets = self.load_raw_datasets()

        train_samples = []
        test_samples = []

        # Sampling and formatting for each dataset
        for dataset_name, dataset in raw_datasets.items():
            print(f"\nSampling for {dataset_name}")

            train_data, test_data = self.sample_dataset(
                dataset,
                self.config.train_samples_per_dataset,
                self.config.test_samples_per_dataset
            )

            # Mark dataset source
            train_data = train_data.add_column(
                "source_dataset",
                [dataset_name] * len(train_data)
            )
            test_data = test_data.add_column(
                "source_dataset",
                [dataset_name] * len(test_data)
            )

            # Convert each dataset to common messages format
            print(f"Formatting {dataset_name}")

            # Map and remove old fields - keep only messages and source_dataset
            all_columns = train_data.column_names
            columns_to_remove = [col for col in all_columns if col != "source_dataset"]

            if dataset_name == "alpaca":
                train_data = train_data.map(
                    self.format_alpaca_sample,
                    remove_columns=columns_to_remove,
                    desc="Formatting alpaca train"
                )
                test_data = test_data.map(
                    self.format_alpaca_sample,
                    remove_columns=columns_to_remove,
                    desc="Formatting alpaca test"
                )
            elif dataset_name == "tulu":
                train_data = train_data.map(
                    self.format_tulu_sample,
                    remove_columns=columns_to_remove,
                    desc="Formatting tulu train"
                )
                test_data = test_data.map(
                    self.format_tulu_sample,
                    remove_columns=columns_to_remove,
                    desc="Formatting tulu test"
                )
            elif dataset_name == "ultrachat":
                train_data = train_data.map(
                    self.format_ultrachat_sample,
                    remove_columns=columns_to_remove,
                    desc="Formatting ultrachat train"
                )
                test_data = test_data.map(
                    self.format_ultrachat_sample,
                    remove_columns=columns_to_remove,
                    desc="Formatting ultrachat test"
                )

            train_samples.append(train_data)
            test_samples.append(test_data)

            print(f"{dataset_name} - Train: {len(train_data)}, Test: {len(test_data)}")

        # Combine all datasets
        print("\nCombining datasets")

        # Check schema of each dataset
        print("\nDataset schema check (after formatting):")
        for i, (train_data, test_data) in enumerate(zip(train_samples, test_samples)):
            dataset_name = list(raw_datasets.keys())[i]
            print(f"\n{dataset_name}:")
            print(f"  Features: {train_data.features}")
            print(f"  First sample keys: {list(train_data[0].keys())}")

            # Check messages field
            if "messages" in train_data[0]:
                print(f"  Messages type: {type(train_data[0]['messages'])}")
                if isinstance(train_data[0]['messages'], list) and len(train_data[0]['messages']) > 0:
                    print(f"  First message keys: {list(train_data[0]['messages'][0].keys())}")
                    print(f"  Number of messages: {len(train_data[0]['messages'])}")

        # Now all datasets are in same format (have messages field)
        # But convert to dict lists to prevent Arrow schema mismatches
        print("\nConverting datasets to dict format and combining")

        train_dicts = []
        test_dicts = []

        for train_data, test_data in zip(train_samples, test_samples):
            # Convert each dataset to dict list
            for sample in train_data:
                # Recreate messages in consistent format
                normalized_messages = []
                for msg in sample["messages"]:
                    normalized_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })

                train_dicts.append({
                    "messages": normalized_messages,
                    "source_dataset": sample["source_dataset"]
                })

            for sample in test_data:
                # Recreate messages in consistent format
                normalized_messages = []
                for msg in sample["messages"]:
                    normalized_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })

                test_dicts.append({
                    "messages": normalized_messages,
                    "source_dataset": sample["source_dataset"]
                })

        # Create new dataset from dict lists with unified schema
        combined_train = Dataset.from_list(train_dicts)
        combined_test = Dataset.from_list(test_dicts)

        # Shuffle
        combined_train = combined_train.shuffle(seed=self.random_seed)
        combined_test = combined_test.shuffle(seed=self.random_seed)

        print(f"\nTotal Train: {len(combined_train)}")
        print(f"\Total Test: {len(combined_test)}")

        return combined_train, combined_test

    def save_datasets(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        output_dir: str = None
    ):
        """
        Saves datasets to disk

        Args:
            train_dataset: Train dataset
            test_dataset: Test dataset
            output_dir: Save directory
        """
        if output_dir is None:
            output_dir = self.config.samples_dir

        os.makedirs(output_dir, exist_ok=True)

        train_path = os.path.join(output_dir, "train_samples.json")
        test_path = os.path.join(output_dir, "test_samples.json")

        print(f"\nSaving datasets")
        print(f"Train: {train_path}")
        print(f"Test: {test_path}")

        # Save in JSON format
        train_dataset.to_json(train_path, orient="records", lines=True)
        test_dataset.to_json(test_path, orient="records", lines=True)

        # Save statistics
        stats = {
            "total_train_samples": len(train_dataset),
            "total_test_samples": len(test_dataset),
            "train_samples_per_dataset": self.config.train_samples_per_dataset,
            "test_samples_per_dataset": self.config.test_samples_per_dataset,
            "random_seed": self.random_seed,
            "datasets": {
                "alpaca": self.config.alpaca_dataset,
                "tulu": self.config.tulu_dataset,
                "ultrachat": self.config.ultrachat_dataset
            }
        }

        stats_path = os.path.join(output_dir, "dataset_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        print(f"Statistics: {stats_path}")
        print("Save completed")


def main():
    """Main function"""
    print("="*50)
    print("Dataset Preparation Started")
    print("="*50)

    preparer = DatasetPreparer()

    # Prepare datasets
    train_dataset, test_dataset = preparer.prepare_all_datasets()

    # Save
    preparer.save_datasets(train_dataset, test_dataset)

    print("\n" + "="*50)
    print("Dataset Preparation Completed")
    print("="*50)


if __name__ == "__main__":
    main()

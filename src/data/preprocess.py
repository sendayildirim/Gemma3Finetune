"""
Dataset preprocessing and formatting
Converts each dataset's own format to Gemma chat template format
"""

import os
import json
from typing import Dict, List, Any
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config.data_config import get_data_config


class DatasetPreprocessor:
    """Class for converting datasets to Gemma chat format"""

    def __init__(self, config=None):
        self.config = config if config else get_data_config()

        # Load Gemma tokenizer
        print("Loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        # Set up chat template for Gemma
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Tokenizer loaded")

    def sample_to_messages(self, sample: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Extracts messages list from sample
        All datasets are already converted to messages format in prepare_datasets.py

        Args:
            sample: Dataset sample

        Returns:
            Messages list
        """
        # All datasets are now in messages format
        messages = sample.get("messages", [])

        # Check if messages is valid
        if not messages or not isinstance(messages, list):
            return []

        # Normalize messages
        normalized_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "").strip()

                if role in ["user", "assistant"] and content:
                    normalized_messages.append({
                        "role": role,
                        "content": content
                    })

        return normalized_messages

    def format_with_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """
        Formats messages with Gemma chat template

        Args:
            messages: Messages list

        Returns:
            Formatted text
        """
        try:
            # Use tokenizer's chat template
            formatted_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            return formatted_text
        except Exception as e:
            print(f"Chat template error: {e}")
            return ""

    def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        """
        Preprocesses entire dataset

        Args:
            dataset: Raw dataset

        Returns:
            Preprocessed dataset
        """
        def process_function(example):
            # Convert to messages format
            messages = self.sample_to_messages(example)

            # Format with chat template
            if messages:
                formatted_text = self.format_with_chat_template(messages)
            else:
                formatted_text = ""

            return {
                "text": formatted_text,
                "source_dataset": example.get("source_dataset", "unknown")
            }

        processed = dataset.map(
            process_function,
            remove_columns=[col for col in dataset.column_names if col != "source_dataset"],
            desc="Preprocessing"
        )

        # Filter empty texts
        processed = processed.filter(lambda x: len(x["text"]) > 0)

        return processed

    def balance_datasets(
        self,
        dataset: Dataset,
        target_per_source: int,
        split_name: str = "train"
    ) -> Dataset:
        """
        Takes equal number of samples from each source_dataset

        Args:
            dataset: Preprocessed dataset
            target_per_source: Number of samples to take from each source
            split_name: "train" or "test" (for logging)

        Returns:
            Balanced dataset
        """
        balanced_samples = []

        for source in ["alpaca", "tulu", "ultrachat"]:
            source_samples = dataset.filter(
                lambda x: x["source_dataset"] == source
            )

            available = len(source_samples)
            print(f"{source} ({split_name}): {available} available, need {target_per_source}")

            if available < target_per_source:
                raise ValueError(
                    f"Not enough {source} samples after filtering! "
                    f"Available: {available}, Need: {target_per_source}. "
                    f"Increase buffer in data_config.py"
                )

            # Take first target_per_source samples
            selected = source_samples.select(range(target_per_source))
            balanced_samples.append(selected)

        # Combine and shuffle
        balanced = concatenate_datasets(balanced_samples)
        balanced = balanced.shuffle(seed=self.config.seed)

        return balanced

    def load_and_preprocess(
        self,
        samples_dir: str = None
    ) -> tuple[Dataset, Dataset]:
        """
        Loads samples and preprocesses them

        Args:
            samples_dir: Directory containing samples

        Returns:
            (train_dataset, test_dataset) tuple
        """
        if samples_dir is None:
            samples_dir = self.config.samples_dir

        train_path = os.path.join(samples_dir, "train_samples.json")
        test_path = os.path.join(samples_dir, "test_samples.json")

        print("Loading samples")
        train_dataset = load_dataset("json", data_files=train_path, split="train")
        test_dataset = load_dataset("json", data_files=test_path, split="train")

        print(f"Train samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")

        print("\nStarting preprocessing")
        train_processed = self.preprocess_dataset(train_dataset)
        test_processed = self.preprocess_dataset(test_dataset)

        print(f"\nPreprocessed train samples: {len(train_processed)}")
        print(f"Preprocessed test samples: {len(test_processed)}")

        # Balance datasets - take equal number from each source
        print("\nBalancing datasets...")
        train_balanced = self.balance_datasets(
            train_processed,
            self.config.target_train_per_dataset,
            "train"
        )
        test_balanced = self.balance_datasets(
            test_processed,
            self.config.target_test_per_dataset,
            "test"
        )

        print(f"\nFinal balanced train samples: {len(train_balanced)}")
        print(f"Final balanced test samples: {len(test_balanced)}")

        return train_balanced, test_balanced

    def save_processed_datasets(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        output_dir: str = None
    ):
        """
        Saves preprocessed datasets

        Args:
            train_dataset: Preprocessed train dataset
            test_dataset: Preprocessed test dataset
            output_dir: Output directory
        """
        if output_dir is None:
            output_dir = self.config.processed_data_dir

        os.makedirs(output_dir, exist_ok=True)

        train_path = os.path.join(output_dir, "train_processed.json")
        test_path = os.path.join(output_dir, "test_processed.json")

        print(f"\nSaving processed datasets")
        print(f"Train: {train_path}")
        print(f"Test: {test_path}")

        train_dataset.to_json(train_path, orient="records", lines=True)
        test_dataset.to_json(test_path, orient="records", lines=True)

        # Save examples
        examples_dir = os.path.join(output_dir, "examples")
        os.makedirs(examples_dir, exist_ok=True)

        for source in ["alpaca", "tulu", "ultrachat"]:
            source_examples = train_dataset.filter(
                lambda x: x["source_dataset"] == source
            )

            if len(source_examples) > 0:
                example_path = os.path.join(examples_dir, f"{source}_example.txt")
                with open(example_path, "w", encoding="utf-8") as f:
                    f.write(source_examples[0]["text"])
                print(f"{source} example: {example_path}")

        print("Save completed")


def main():
    """Main function"""
    print("="*50)
    print("Dataset Preprocessing Started")
    print("="*50)

    preprocessor = DatasetPreprocessor()

    # Load and preprocess
    train_dataset, test_dataset = preprocessor.load_and_preprocess()

    # Save
    preprocessor.save_processed_datasets(train_dataset, test_dataset)

    print("\n" + "="*50)
    print("Preprocessing Completed")
    print("="*50)


if __name__ == "__main__":
    main()

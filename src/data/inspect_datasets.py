"""
Dataset structure inspection and debug utility
Shows fields and examples of each dataset
"""

from datasets import load_dataset
import json


def inspect_dataset(dataset_name: str, dataset_path: str, split: str = "train", n_samples: int = 2):
    """
    Inspects dataset structure

    Args:
        dataset_name: Dataset name
        dataset_path: HuggingFace dataset path
        split: Split name
        n_samples: Number of examples to show
    """
    print("\n" + "="*60)
    print(f"Dataset: {dataset_name}")
    print(f"Path: {dataset_path}")
    print("="*60)

    try:
        dataset = load_dataset(dataset_path, split=split)

        print(f"\nTotal samples: {len(dataset)}")
        print(f"\nFields: {dataset.column_names}")
        print(f"\nFeatures: {dataset.features}")

        print(f"\n{n_samples} Examples:")
        print("-"*60)

        for i in range(min(n_samples, len(dataset))):
            sample = dataset[i]
            print(f"\nExample {i+1}:")
            print(json.dumps(sample, indent=2, ensure_ascii=False))
            print("-"*60)

    except Exception as e:
        print(f"Error: {e}")


def main():
    """Main function - inspect all datasets"""

    datasets_to_inspect = [
        ("Alpaca", "tatsu-lab/alpaca", "train"),
        ("Tulu v2 SFT", "allenai/tulu-v2-sft-mixture", "train"),
        ("Ultrachat 200k", "HuggingFaceH4/ultrachat_200k", "train_sft"),
    ]

    for name, path, split in datasets_to_inspect:
        inspect_dataset(name, path, split, n_samples=2)

    print("\n" + "="*60)
    print("Inspection Completed")
    print("="*60)


if __name__ == "__main__":
    main()

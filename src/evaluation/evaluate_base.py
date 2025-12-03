"""
Base model evaluation script
Evaluates the Gemma-3-1b-it base model on test set
"""

import os
import sys
import json
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from evaluate import load
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config.data_config import get_data_config


class BaseModelEvaluator:
    """Base model evaluation class"""

    def __init__(self, model_name: str = "google/gemma-3-1b-it"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.bleu_metric = load("bleu")
        self.rouge_metric = load("rouge")

    def load_model(self):
        """Loads model and tokenizer"""
        print(f"Loading base model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )

        print("Model loaded")

    def extract_instruction_and_response(self, text: str):
        """
        Extracts instruction and response from formatted text

        Args:
            text: Formatted chat text

        Returns:
            (instruction, expected_response) tuple
        """
        # Gemma format: <start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn>
        try:
            parts = text.split("<start_of_turn>")

            instruction = ""
            expected_response = ""

            for part in parts:
                if part.startswith("user\n"):
                    instruction = part.replace("user\n", "").replace("<end_of_turn>", "").strip()
                elif part.startswith("model\n"):
                    expected_response = part.replace("model\n", "").replace("<end_of_turn>", "").strip()

            return instruction, expected_response

        except Exception as e:
            print(f"Parse error: {e}")
            return "", ""

    def generate_response(self, instruction: str, max_new_tokens: int = 256) -> str:
        """
        Generates model response for instruction

        Args:
            instruction: User instruction
            max_new_tokens: Maximum number of tokens

        Returns:
            Generated response
        """
        messages = [{"role": "user", "content": instruction}]

        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove prompt, keep only generated part
        response = generated_text.replace(prompt, "").strip()

        return response

    def generate_batch_responses(self, instructions: list, max_new_tokens: int = 256, batch_size: int = 64) -> list:
        """
        Generates parallel responses for batch instructions

        Args:
            instructions: Instruction list
            max_new_tokens: Maximum number of tokens
            batch_size: Batch size

        Returns:
            List of generated responses
        """
        all_responses = []

        for i in range(0, len(instructions), batch_size):
            batch_instructions = instructions[i:i+batch_size]

            # Create prompt for each instruction
            batch_prompts = []
            for instruction in batch_instructions:
                messages = [{"role": "user", "content": instruction}]
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                batch_prompts.append(prompt)

            # Batch tokenization with padding
            inputs = self.tokenizer(
                batch_prompts,
                padding=True,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.model.device)

            # Batch generation
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            # Decode batch
            batch_responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # Remove prompts
            for prompt, generated_text in zip(batch_prompts, batch_responses):
                response = generated_text.replace(prompt, "").strip()
                all_responses.append(response)

        return all_responses

    def evaluate_on_test_set(
        self,
        test_dataset_path: str,
        max_samples: int = None,
        output_dir: str = "./results/metrics"
    ):
        """
        Performs evaluation on test set

        Args:
            test_dataset_path: Test dataset JSON path
            max_samples: Maximum number of samples (use all if None)
            output_dir: Output directory
        """
        print("\nLoading test dataset")
        test_dataset = load_dataset("json", data_files=test_dataset_path, split="train")

        if max_samples:
            test_dataset = test_dataset.select(range(min(max_samples, len(test_dataset))))

        print(f"Test dataset size: {len(test_dataset)}")

        # Extract all instructions and references
        instructions = []
        references = []

        print("\nExtracting instructions")
        for sample in test_dataset:
            text = sample["text"]
            instruction, expected_response = self.extract_instruction_and_response(text)

            if not instruction or not expected_response:
                continue

            instructions.append(instruction)
            references.append(expected_response)

        print(f"Found {len(instructions)} valid samples")

        # Batch generation
        print("\nStarting batch generation (batch_size=64)")
        predictions = self.generate_batch_responses(
            instructions,
            max_new_tokens=256,
            batch_size=64
        )

        print("\nCalculating metrics")

        # Calculate BLEU-4
        bleu_results = self.bleu_metric.compute(
            predictions=predictions,
            references=[[ref] for ref in references],
            max_order=4
        )
        bleu_4 = bleu_results["bleu"]

        # Calculate ROUGE-L
        rouge_results = self.rouge_metric.compute(
            predictions=predictions,
            references=references
        )
        rouge_l = rouge_results["rougeL"]

        results = {
            "model": self.model_name,
            "technique": "Base Model",
            "bleu_4": bleu_4,
            "rouge_l": rouge_l,
            "num_samples": len(predictions),
        }

        print("\n" + "="*60)
        print("Base Model Evaluation Results")
        print("="*60)
        print(f"BLEU-4: {bleu_4:.4f}")
        print(f"ROUGE-L: {rouge_l:.4f}")
        print(f"Evaluated samples: {len(predictions)}")

        os.makedirs(output_dir, exist_ok=True)
        results_file = os.path.join(output_dir, "base_model_results.json")

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved: {results_file}")

        # Save examples
        examples_file = os.path.join(output_dir, "base_model_examples.json")
        examples = []
        for i in range(min(10, len(predictions))):
            examples.append({
                "instruction": self.extract_instruction_and_response(test_dataset[i]["text"])[0],
                "expected": references[i],
                "generated": predictions[i]
            })

        with open(examples_file, "w", encoding="utf-8") as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)

        print(f"Examples saved: {examples_file}")

        return results


def main():
    """Main function"""
    print("="*60)
    print("Base Model Evaluation")
    print("="*60)

    config = get_data_config()
    evaluator = BaseModelEvaluator(model_name=config.model_name)

    evaluator.load_model()

    test_dataset_path = os.path.join(config.processed_data_dir, "test_processed.json")

    # Use a portion of test set (for speed)
    evaluator.evaluate_on_test_set(
        test_dataset_path,
        max_samples=500
    )

    print("\n" + "="*60)
    print("Base Model Evaluation Completed")
    print("="*60)


if __name__ == "__main__":
    main()

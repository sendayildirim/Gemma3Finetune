"""
Fine-tuned model evaluation script
Evaluates fine-tuned models with QLoRA and GaLore
"""

import os
import sys
import json
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, AutoPeftModelForCausalLM
from evaluate import load
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config.data_config import get_data_config


class FineTunedModelEvaluator:
    """Fine-tuned model evaluation class"""

    def __init__(self, technique: str, model_path: str, base_model_name: str = "google/gemma-3-1b-it"):
        self.technique = technique
        self.model_path = model_path
        self.base_model_name = base_model_name
        self.tokenizer = None
        self.model = None
        self.bleu_metric = load("bleu")
        self.rouge_metric = load("rouge")

    def load_model(self):
        """Loads model and tokenizer"""
        print(f"\nLoading {self.technique} model: {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.technique == "QLoRA":
            # Load PEFT model for QLoRA
            try:
                self.model = AutoPeftModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                )
                print(f"{self.technique} model loaded (PEFT)")
            except Exception as e:
                print(f"PEFT model loading error: {e}")
                print("Trying via base model")
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                )
                self.model = PeftModel.from_pretrained(base_model, self.model_path)

        else:
            # Load normal model for GaLore
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )

        print(f"{self.technique} model loaded")

    def extract_instruction_and_response(self, text: str):
        """
        Extracts instruction and response from formatted text

        Args:
            text: Formatted chat text

        Returns:
            (instruction, expected_response) tuple
        """
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
            max_samples: Maximum number of samples
            output_dir: Output directory
        """
        print("\nLoading test dataset")
        test_dataset = load_dataset("json", data_files=test_dataset_path, split="train")

        if max_samples:
            test_dataset = test_dataset.select(range(min(max_samples, len(test_dataset))))

        print(f"Test dataset size: {len(test_dataset)}")

        # Extract all instructions and references
        instructions_list = []
        references = []

        print("\nExtracting instructions")
        for sample in test_dataset:
            text = sample["text"]
            instruction, expected_response = self.extract_instruction_and_response(text)

            if not instruction or not expected_response:
                continue

            instructions_list.append(instruction)
            references.append(expected_response)

        print(f"Found {len(instructions_list)} valid samples")

        # Batch generation
        print(f"\nStarting {self.technique} batch generation (batch_size=64)")
        predictions = self.generate_batch_responses(
            instructions_list,
            max_new_tokens=256,
            batch_size=64
        )

        print("\nCalculating metrics")

        bleu_results = self.bleu_metric.compute(
            predictions=predictions,
            references=[[ref] for ref in references],
            max_order=4
        )
        bleu_4 = bleu_results["bleu"]

        rouge_results = self.rouge_metric.compute(
            predictions=predictions,
            references=references
        )
        rouge_l = rouge_results["rougeL"]

        # Load training metrics (if available)
        training_metrics_path = os.path.join(output_dir, f"{self.technique.lower()}_metrics.json")
        peak_memory = None
        training_time = None

        if os.path.exists(training_metrics_path):
            with open(training_metrics_path, "r") as f:
                training_metrics = json.load(f)
                peak_memory = training_metrics.get("memory_stats", {}).get("peak_memory_allocated_gb")
                training_time = training_metrics.get("training_time_hours")

        results = {
            "model": self.base_model_name,
            "technique": self.technique,
            "model_path": self.model_path,
            "bleu_4": bleu_4,
            "rouge_l": rouge_l,
            "num_samples": len(predictions),
            "peak_memory_gb": peak_memory,
            "training_time_hours": training_time,
        }

        print("\n" + "="*60)
        print(f"{self.technique} Evaluation Results")
        print("="*60)
        print(f"BLEU-4: {bleu_4:.4f}")
        print(f"ROUGE-L: {rouge_l:.4f}")
        print(f"Evaluated samples: {len(predictions)}")
        if peak_memory:
            print(f"Peak Memory: {peak_memory:.2f} GB")
        if training_time:
            print(f"Training Time: {training_time:.2f} hours")

        os.makedirs(output_dir, exist_ok=True)
        results_file = os.path.join(output_dir, f"{self.technique.lower()}_evaluation_results.json")

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved: {results_file}")

        # Save 5-10 examples
        examples_file = os.path.join(output_dir, f"{self.technique.lower()}_examples.json")
        examples = []
        for i in range(min(10, len(predictions))):
            examples.append({
                "instruction": instructions_list[i],
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
    print("Fine-Tuned Models Evaluation")
    print("="*60)

    config = get_data_config()
    test_dataset_path = os.path.join(config.processed_data_dir, "test_processed.json")

    # QLoRA evaluation
    qlora_model_path = "./models/qlora/final"
    if os.path.exists(qlora_model_path):
        qlora_evaluator = FineTunedModelEvaluator(
            technique="QLoRA",
            model_path=qlora_model_path,
            base_model_name=config.model_name
        )
        qlora_evaluator.load_model()
        qlora_evaluator.evaluate_on_test_set(test_dataset_path, max_samples=500)
    else:
        print(f"QLoRA model not found: {qlora_model_path}")

    # GaLore evaluation
    galore_model_path = "./models/galore/final"
    if os.path.exists(galore_model_path):
        galore_evaluator = FineTunedModelEvaluator(
            technique="GaLore",
            model_path=galore_model_path,
            base_model_name=config.model_name
        )
        galore_evaluator.load_model()
        galore_evaluator.evaluate_on_test_set(test_dataset_path, max_samples=500)
    else:
        print(f"GaLore model not found: {galore_model_path}")

    print("\n" + "="*60)
    print("Fine-Tuned Models Evaluation Completed")
    print("="*60)


if __name__ == "__main__":
    main()

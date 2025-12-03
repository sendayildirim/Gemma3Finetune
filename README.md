# LLM Fine-Tuning: Comparing PEFT Techniques on Gemma-3-1B-IT

This project aims to fine-tune the `google/gemma-3-1b-it` model using two different Parameter-Efficient Fine-Tuning (PEFT) techniques and compare them in terms of performance, resource usage, and generalization.

## Project Goals

- Fine-tune the Gemma-3-1B-IT model using **QLoRA** and **GaLore** techniques
- Perform performance comparison using BLEU-4 and ROUGE-L metrics
- Analyze resource efficiency in terms of memory usage and training time
- Propose a hybrid/ensemble approach combining the two techniques

## Technologies Used

### Fine-Tuning Techniques

**1. QLoRA (Quantized Low-Rank Adaptation)**
- Base model with 4-bit quantization
- Efficient training with LoRA adapters
- High memory efficiency

**2. GaLore (Gradient Low-Rank Projection)**
- Projecting gradients into low-rank subspace
- Adapter-free approach
- Different optimization strategy

### Datasets

A total of **15,000 training** and **6,000 test** samples from:
- **Alpaca** (tatsu-lab/alpaca) - 5,000 train, 2,000 test
- **Tulu v2 SFT** (allenai/tulu-v2-sft-mixture) - 5,000 train, 2,000 test
- **Ultrachat 200k** (HuggingFaceH4/ultrachat_200k) - 5,000 train, 2,000 test

## Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended: 16GB+ VRAM)
- 32GB+ RAM

### Step 1: Clone the Repository

```bash
cd FineTune
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set HuggingFace Token

A HuggingFace token is required to access the Gemma model:

1. Create a token from your [HuggingFace](https://huggingface.co/settings/tokens) account
2. Request access to the Gemma model: [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it)
3. Set the token as an environment variable:

```bash
export HF_TOKEN="your_token_here"
```

## Usage

### Pipeline Steps

The project consists of 4 main steps:

#### 1. Dataset Preparation

Download datasets and perform sampling:

```bash
python src/data/prepare_datasets.py
```

This script:
- Downloads 3 datasets from HuggingFace
- Takes random 5,000 train + 2,000 test samples from each
- Saves to `data/samples/` directory
- Performs train/test overlap check

**Output:**
- `data/samples/train_samples.json`
- `data/samples/test_samples.json`
- `data/samples/dataset_stats.json`

#### 2. Preprocessing

Convert datasets to Gemma chat template format:

```bash
python src/data/preprocess.py
```

This script:
- Converts each dataset's format to unified messages format
- Applies Gemma tokenizer's chat template
- Saves to `data/processed/` directory

**Output:**
- `data/processed/train_processed.json`
- `data/processed/test_processed.json`
- `data/processed/examples/` (example from each dataset)

##### Dataset Field Mapping Details

Each dataset has a different structure and these are converted to unified Gemma chat template format:

**Gemma Chat Template Format:**
```
<start_of_turn>user
{user_message}<end_of_turn>
<start_of_turn>model
{assistant_response}<end_of_turn>
```

**1. Alpaca Dataset Mapping:**

Original format:
- `instruction`: Main task description
- `input`: (Optional) Additional context or input
- `output`: Expected response

Conversion:
```python
# If input exists:
user_message = f"{instruction}\n\nInput: {input}"
# If no input:
user_message = instruction

messages = [
    {"role": "user", "content": user_message},
    {"role": "assistant", "content": output}
]
```

**2. Tulu v2 SFT Dataset Mapping:**

Original format:
- `messages`: Already in conversation format (list of dicts)
- Each message: `{"role": "...", "content": "..."}`

Conversion:
```python
# Role normalization:
# "user" or "human" → "user"
# "assistant" or "gpt" → "assistant"
# "system" messages are added to user message

messages = normalize_roles(original_messages)
```

**3. Ultrachat 200k Dataset Mapping:**

Original format:
- `messages`: Conversation format (list of dicts)
- Each message: `{"content": "...", "role": "..."}`

Conversion:
```python
# Role normalization and filtering:
# "user" → "user"
# "assistant" → "assistant"
# Other roles are filtered

# Take turns that start with user and end with assistant
messages = normalize_and_filter(original_messages)
```

**Chat Template Application:**

After all datasets are converted to messages format, the final format is created using Gemma tokenizer's `apply_chat_template` function:

```python
formatted_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False
)
```

This process converts each example to Gemma's native chat template format and prepares it for model training.

#### 3. Fine-Tuning

**QLoRA Training:**

```bash
python src/training/train_qlora.py
```

**GaLore Training:**

```bash
python src/training/train_galore.py
```

Each training script:
- Loads model and tokenizer
- Performs fine-tuning
- Tracks memory and time metrics
- Saves model checkpoints

**Output:**
- `models/qlora/final/` - QLoRA fine-tuned model
- `models/galore/final/` - GaLore fine-tuned model
- `results/metrics/qlora_metrics.json` - QLoRA training metrics
- `results/metrics/galore_metrics.json` - GaLore training metrics

#### 4. Evaluation

**Base Model Evaluation:**

```bash
python src/evaluation/evaluate_base.py
```

**Fine-Tuned Models Evaluation:**

```bash
python src/evaluation/evaluate_models.py
```

**Visualizing Results:**

```bash
python src/evaluation/visualize_results.py
```

**Output:**
- `results/metrics/base_model_results.json`
- `results/metrics/qlora_evaluation_results.json`
- `results/metrics/galore_evaluation_results.json`
- `results/metrics/comparison_table.csv`
- `results/plots/bleu_rouge_comparison.png`
- `results/plots/memory_vs_performance.png`
- `results/summary_report.txt`

## Project Structure

```
FineTune/
├── README.md
├── requirements.txt
├── src/
│   ├── config/
│   │   ├── data_config.py          # Dataset configuration
│   │   ├── qlora_config.py         # QLoRA hyperparameters
│   │   └── galore_config.py        # GaLore hyperparameters
│   ├── data/
│   │   ├── prepare_datasets.py     # Dataset sampling
│   │   ├── preprocess.py           # Preprocessing and formatting
│   │   └── inspect_datasets.py     # Dataset structure inspection
│   ├── training/
│   │   ├── train_qlora.py          # QLoRA fine-tuning
│   │   ├── train_galore.py         # GaLore fine-tuning
│   │   └── utils.py                # Training utilities
│   └── evaluation/
│       ├── evaluate_base.py        # Base model evaluation
│       ├── evaluate_models.py      # Fine-tuned models evaluation
│       └── visualize_results.py    # Results visualization
├── data/
│   ├── raw/                        # Raw datasets
│   ├── samples/                    # Sampled datasets
│   └── processed/                  # Preprocessed datasets
├── models/
│   ├── qlora/                      # QLoRA checkpoints
│   └── galore/                     # GaLore checkpoints
├── results/
│   ├── metrics/                    # Evaluation metrics
│   ├── plots/                      # Graphs
│   └── examples/                   # Example generations
├── notebooks/
│   └── finetune_gemma_colab.ipynb # Google Colab notebook
└── docs/
    ├── report.md                   # Final report
    └── hybrid_proposal.md          # Ensemble/Hybrid approach
```

## Configuration

To modify hyperparameters, edit the config files:

### QLoRA Config (`src/config/qlora_config.py`)

```python
lora_r = 16                    # LoRA rank
lora_alpha = 32                # LoRA alpha
lora_dropout = 0.05            # Dropout rate
learning_rate = 2e-4           # Learning rate
num_train_epochs = 3           # Training epochs
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
```

### GaLore Config (`src/config/galore_config.py`)

```python
rank = 128                     # GaLore rank
update_proj_gap = 200          # Projection update interval
galore_scale = 0.25            # Scaling factor
learning_rate = 1e-4           # Learning rate
num_train_epochs = 3           # Training epochs
```

## Results Summary

For detailed results, see `results/summary_report.txt`.

### Comparison Table

| Technique | BLEU-4 (Before) | BLEU-4 (After) | ROUGE-L (Before) | ROUGE-L (After) | Peak Memory (GB) | Training Time (Hrs) |
|-----------|-----------------|----------------|------------------|-----------------|------------------|---------------------|
| QLoRA     | 0.0582          | 0.1112         | 0.1632           | 0.2326          | 1.51             | 2.08                |
| GaLore    | 0.0582          | 0.1064         | 0.1632           | 0.2268          | 2.95             | 1.40                |

**Key Findings:**
- QLoRA achieved 91% improvement in BLEU-4 and 43% improvement in ROUGE-L over the base model
- GaLore achieved 83% improvement in BLEU-4 and 39% improvement in ROUGE-L over the base model
- QLoRA uses 49% less memory than GaLore but requires 33% more training time
- Both techniques successfully improved model performance with efficient resource usage

## Hybrid/Ensemble Approach

For detailed explanation of the proposed hybrid approach: [docs/hybrid_proposal.md](docs/hybrid_proposal.md)

## Using Google Colab

To run the entire pipeline in Google Colab:

1. Open the `notebooks/finetune_gemma_colab.ipynb` file
2. Upload to your Google Drive
3. Using Colab Pro is recommended (for GPU and high RAM)
4. Add your HuggingFace token to Colab Secrets
5. Run the notebook sequentially

## Troubleshooting

### CUDA Out of Memory

- Reduce batch size
- Increase gradient accumulation steps
- Reduce `max_seq_length`

### HuggingFace Token Error

- Ensure token is set correctly
- Verify you have access permission for Gemma model

### Dataset Download Slow

- Check your internet connection
- Clean HuggingFace cache directory: `~/.cache/huggingface/`

## Contributing

This project is an academic assessment project.

## License

This project is for educational purposes.

## Resources

- [Gemma Model](https://huggingface.co/google/gemma-3-1b-it)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [GaLore Paper](https://arxiv.org/abs/2403.03507)
- [HuggingFace PEFT](https://github.com/huggingface/peft)
- [TRL Library](https://github.com/huggingface/trl)

## Contact

You can open an issue for questions.

**Final Report:** [docs/report.md](docs/report.md)

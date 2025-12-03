# Fine-Tuning Gemma-3-1B with QLoRA and GaLore: A Comparative Analysis

## Executive Summary

This report presents a comprehensive comparison of two Parameter-Efficient Fine-Tuning (PEFT) techniques—**QLoRA (Quantized Low-Rank Adaptation)** and **GaLore (Gradient Low-Rank Projection)**—applied to the `google/gemma-3-1b-it` language model. The study evaluates their effectiveness across three key dimensions: performance metrics (BLEU-4, ROUGE-L), resource consumption (memory, training time), and generalization behavior on instruction-following tasks.

**Key Findings:**
- QLoRA achieved BLEU-4: 0.1112 and ROUGE-L: 0.2326
- GaLore achieved BLEU-4: 0.1064 and ROUGE-L: 0.2268
- QLoRA peak memory usage: 1.51 GB
- GaLore peak memory usage: 2.95 GB
- QLoRA training time: 2.08 hours
- GaLore training time: 1.40 hours

---

## 1. Objective

The primary goal of this study is to fine-tune the `google/gemma-3-1b-it` large language model using two distinct Parameter-Efficient Fine-Tuning (PEFT) techniques and compare their effectiveness based on:

1. **Performance metrics**: BLEU-4 and ROUGE-L scores on a held-out test set
2. **Resource consumption**: GPU memory usage and training time
3. **Generalization behavior**: Analysis of model responses to unseen instructions

---

## 2. Base Language Model

**Model:** `google/gemma-3-1b-it`
**Link:** https://huggingface.co/google/gemma-3-1b-it

### Rationale

The Gemma-3-1B instruction-tuned model was selected for the following reasons:

1. **Strong Baseline Performance**: Pre-trained on instruction-following tasks, providing a robust starting point for fine-tuning
2. **Resource Feasibility**: At 3.1B parameters, the model is large enough to demonstrate meaningful improvements from PEFT techniques while remaining feasible for experimentation within typical resource constraints (Google Colab, consumer GPUs)
3. **Native Chat Template Support**: Built-in chat template (`<start_of_turn>` format) simplifies preprocessing and ensures optimal model performance
4. **Architecture**: Transformer-based architecture with attention and MLP layers suitable for both adapter-based (QLoRA) and gradient-based (GaLore) optimization techniques

---

## 3. Datasets & Sampling

### Dataset Selection

Three high-quality instruction-following datasets were selected to create a diverse training corpus:

1. **Alpaca** (`tatsu-lab/alpaca`)
   - General instruction-following dataset
   - Format: instruction/input/output

2. **Tulu v2 SFT** (`allenai/tulu-v2-sft-mixture`)
   - Multi-task mixture dataset
   - Format: messages (conversation)

3. **Ultrachat 200k** (`HuggingFaceH4/ultrachat_200k`)
   - High-quality conversational dataset
   - Format: messages (multi-turn dialogue)

### Sampling Strategy

To ensure data quality and prevent train/test contamination:

**Per-Dataset Sampling:**
- **Training samples**: 5,000 per dataset (15,000 total)
- **Test samples**: 2,000 per dataset (6,000 total)

**Process:**
1. Random sampling with `seed=42` from each dataset independently
2. Non-overlapping train/test split: indices are completely separate within each dataset
3. Verification through assertion: `assert len(set(train_indices) & set(test_indices)) == 0`
4. Two-stage shuffling:
   - First shuffle: During sampling from each source dataset
   - Second shuffle: After combining all three datasets
5. Source tracking: Each sample tagged with `source_dataset` field

**Justification:**
- Equal representation from all three datasets prevents domain bias
- Non-overlapping splits ensure valid evaluation
- Reproducible sampling (seed=42) enables experiment replication
- Sufficient size (15K train, 6K test) balances training effectiveness with computational feasibility

---

## 4. Data Preprocessing

All three datasets were converted to a unified format using the Gemma native chat template to ensure consistency and optimal model performance.

### Gemma Chat Template Format

```
<start_of_turn>user
{user_message}<end_of_turn>
<start_of_turn>model
{assistant_response}<end_of_turn>
```

### Dataset-Specific Transformations

#### 4.1 Alpaca Dataset

**Original Format:**
```json
{
  "instruction": "Summarize the given article",
  "input": "Article text here",
  "output": "Summary here"
}
```

**Transformation Logic:**
```python
def convert_alpaca_to_messages(sample):
    instruction = sample.get("instruction", "").strip()
    input_text = sample.get("input", "").strip()
    output_text = sample.get("output", "").strip()

    # Combine instruction and input if input exists
    if input_text:
        user_content = f"{instruction}\n\nInput: {input_text}"
    else:
        user_content = instruction

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output_text}
    ]
    return messages
```

**Resulting Format:**
```
<start_of_turn>user
Summarize the given article

Input: Article text here<end_of_turn>
<start_of_turn>model
Summary here<end_of_turn>
```

#### 4.2 Tulu v2 SFT Dataset

**Original Format:**
```json
{
  "messages": [
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is"}
  ]
}
```

**Transformation Logic:**
```python
def convert_tulu_to_messages(sample):
    messages = sample.get("messages", [])
    normalized = []

    for msg in messages:
        role = msg.get("role", "").lower()
        content = msg.get("content", "").strip()

        # Normalize role names
        if role in ["user", "human"]:
            normalized.append({"role": "user", "content": content})
        elif role in ["assistant", "gpt"]:
            normalized.append({"role": "assistant", "content": content})
        elif role == "system":
            # Prepend system message to first user message
            if normalized and normalized[-1]["role"] == "user":
                normalized[-1]["content"] = f"{content}\n\n{normalized[-1]['content']}"

    return normalized
```

#### 4.3 Ultrachat 200k Dataset

**Original Format:**
```json
{
  "messages": [
    {"role": "user", "content": "Explain quantum computing"},
    {"role": "assistant", "content": "Quantum computing uses"}
  ]
}
```

**Transformation Logic:**
```python
def convert_ultrachat_to_messages(sample):
    messages = sample.get("messages", [])
    normalized = []

    for msg in messages:
        role = msg.get("role", "").lower()
        content = msg.get("content", "").strip()

        # Filter and normalize
        if role == "user":
            normalized.append({"role": "user", "content": content})
        elif role == "assistant":
            normalized.append({"role": "assistant", "content": content})

    # Ensure conversation starts with user and ends with assistant
    if normalized and normalized[0]["role"] == "user" and normalized[-1]["role"] == "assistant":
        return normalized
    return []
```

### Final Template Application

After converting all datasets to the unified `messages` format, the Gemma tokenizer's chat template is applied:

```python
formatted_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False
)
```

### Preprocessing Verification

Examples from each dataset after preprocessing are saved to `data/processed/examples/` for manual inspection:
- `alpaca_example.txt`
- `tulu_example.txt`
- `ultrachat_example.txt`

### Data Quality Measures

1. **Empty content filtering**: Samples with empty user or assistant content are removed
2. **Format validation**: Conversations must start with user and end with assistant
3. **Length constraints**: Maximum sequence length of 2048 tokens
4. **Character encoding**: UTF-8 encoding ensures proper handling of special characters

### Over-Sampling and Dataset Balancing Strategy

To ensure exactly 5,000 train and 2,000 test samples per dataset after preprocessing losses, a two-stage approach was implemented:

**Stage 1: Over-Sampling (prepare_datasets.py)**

Initial sampling with buffer to account for preprocessing losses:
```python
train_samples_per_dataset = 5200  # 4% buffer (200 extra samples)
test_samples_per_dataset = 2100   # 5% buffer (100 extra samples)
```

**Rationale for Buffer Size:**
- Gemma chat template requires strict alternating user/assistant roles
- Some samples have consecutive messages from same role (e.g., user→user or assistant→assistant)
- These invalid samples are automatically filtered during chat template application
- Empirical testing showed ~0.1-0.5% loss rate, but 4-5% buffer ensures safety margin

**Stage 2: Balancing (preprocess.py)**

After chat template application and filtering, exactly the target number of samples is selected from each dataset:

```python
def balance_datasets(dataset, target_per_source, split_name):
    balanced_samples = []

    for source in ["alpaca", "tulu", "ultrachat"]:
        source_samples = dataset.filter(
            lambda x: x["source_dataset"] == source
        )

        available = len(source_samples)
        print(f"{source} ({split_name}): {available} available, need {target_per_source}")

        if available < target_per_source:
            raise ValueError(
                f"Not enough {source} samples after filtering "
                f"Available: {available}, Need: {target_per_source}. "
                f"Increase buffer in data_config.py"
            )

        # Select first target_per_source samples
        selected = source_samples.select(range(target_per_source))
        balanced_samples.append(selected)

    # Concatenate and shuffle
    balanced = concatenate_datasets(balanced_samples)
    balanced = balanced.shuffle(seed=42)

    return balanced

# Apply balancing
train_balanced = balance_datasets(train_processed, 5000, "train")
test_balanced = balance_datasets(test_processed, 2000, "test")
```

**Key Benefits:**

1. **Guaranteed Equal Representation**: Exactly 5,000 train and 2,000 test samples from each dataset source
2. **No Data Loss**: Buffer ensures sufficient samples survive filtering
3. **Fail-Safe Mechanism**: Raises error if buffer insufficient, rather than silently producing imbalanced data
4. **Reproducibility**: Deterministic selection (first N samples) and shuffling (seed=42)

**Final Dataset Composition:**

| Dataset | Train Samples | Test Samples | Total |
|---------|--------------|--------------|-------|
| Alpaca | 5,000 | 2,000 | 7,000 |
| Tulu v2 | 5,000 | 2,000 | 7,000 |
| Ultrachat | 5,000 | 2,000 | 7,000 |
| **Total** | **15,000** | **6,000** | **21,000** |

This approach ensures balanced representation from all three data sources, preventing domain bias and ensuring the model learns diverse instruction-following patterns.

---

## 5. Fine-Tuning Techniques

### 5.1 Selected Techniques

Two distinct PEFT techniques were chosen to represent different approaches to parameter efficiency:

#### **Technique 1: QLoRA (Quantized Low-Rank Adaptation)**

**Approach:** Adapter-based PEFT with quantization
**Libraries:** `transformers`, `peft`, `bitsandbytes`, `trl`

**Key Characteristics:**
- 4-bit quantization (NF4) of base model weights
- Low-rank adapter matrices (LoRA) applied to attention and MLP layers
- Only adapter weights are trainable (frozen quantized base)
- Memory-efficient through quantization + parameter reduction

**Implementation Details:**
```python
# Quantization Configuration
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# LoRA Configuration
LoraConfig(
    r=16,                    # Rank of adapter matrices
    lora_alpha=32,           # Scaling factor
    lora_dropout=0.05,       # Dropout for regularization
    target_modules=[         # Gemma attention/MLP layers
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    bias="none",
    task_type="CAUSAL_LM"
)
```

**Trainable Parameters:**
- Total parameters: ~664M (quantized to 4-bit)
- Trainable parameters: 13,045,760 (1.96% of total)

#### **Technique 2: GaLore (Gradient Low-Rank Projection)**

**Approach:** Gradient-based optimization in low-rank subspace
**Libraries:** `transformers`, `galore_torch`, `trl`

**Key Characteristics:**
- Full-precision base model (bfloat16)
- Gradient projection into low-rank subspace during backpropagation
- All model parameters updated (not adapter-based)
- Memory savings through gradient compression

**Implementation Details:**
```python
# GaLore Optimizer Configuration
GaLoreAdamW(
    model.parameters(),
    lr=1e-4,
    rank=128,                # Rank of gradient projection
    update_proj_gap=200,     # Steps between projection updates
    scale=0.25,              # Gradient scaling factor
    proj_type="std"          # Standard projection type
)

# Target Layers
target_modules_list = ["attn", "mlp"]  # Apply to attention and MLP layers
```

**Trainable Parameters:**
- Total parameters: ~1B (full precision bfloat16)
- Trainable parameters: 999,885,952 (100% of total - all parameters updated)

### 5.2 Training Configuration

Both techniques were trained with **identical hyperparameters** wherever possible to ensure fair comparison:

| Hyperparameter | QLoRA | GaLore | Rationale |
|----------------|-------|---------|-----------|
| **Epochs** | 3 | 3 | Sufficient for convergence |
| **Batch Size (per device)** | 4 | 4 | Memory constraint |
| **Gradient Accumulation** | 4 | 4 | Effective batch size = 16 |
| **Learning Rate** | 2e-4 | 1e-4 | Adjusted for technique |
| **Weight Decay** | 0.01 | 0.01 | Regularization |
| **Warmup Steps** | 100 | 100 | Learning rate warmup |
| **Max Sequence Length** | 2048 | 2048 | Gemma context window |
| **Optimizer** | PagedAdamW 8-bit | GaLoreAdamW | Technique-specific |
| **LR Scheduler** | Cosine | Cosine | Smooth decay |
| **Precision** | BF16 | BF16 | Numerical stability |
| **Gradient Checkpointing** | Yes | Yes | Memory optimization |
| **Max Grad Norm** | 0.3 | 1.0 | Gradient clipping |
| **Random Seed** | 42 | 42 | Reproducibility |

**Differences Explained:**
- **Learning Rate**: QLoRA uses 2e-4 (typical for adapter training), GaLore uses 1e-4 (more conservative for full-model gradient updates)
- **Optimizer**: QLoRA uses paged 8-bit AdamW (memory efficiency), GaLore uses custom GaLoreAdamW (gradient projection)
- **Max Grad Norm**: QLoRA uses 0.3 (adapters sensitive to large gradients), GaLore uses 1.0 (full model more stable)

### 5.3 Justification for Technique Selection

#### **QLoRA Justification**

**Performance:**
- **Proven effectiveness**: Extensively validated on instruction-following benchmarks
- **Preserved base model knowledge**: Frozen quantized weights retain pre-training
- **Efficient adaptation**: Low-rank adapters capture task-specific patterns effectively
- **Expected outcome**: Strong performance with minimal trainable parameters

**Memory Efficiency:**
- **4-bit quantization**: Reduces base model memory by ~75% (32-bit → 4-bit)
- **Small adapter footprint**: Only ~1.96% additional trainable parameters
- **Optimizer states**: 8-bit paged optimizer reduces memory further
- **Actual memory**: 1.51 GB peak GPU memory

**Generalization:**
- **Selective updating**: Only task-relevant parameters (adapters) modified
- **Preserved capabilities**: Base model knowledge intact for broader tasks
- **Risk mitigation**: Less prone to catastrophic forgetting
- **Hypothesis**: Should generalize well to instruction variations within training distribution

#### **GaLore Justification**

**Performance:**
- **Full model updates**: All parameters can adapt to new task distribution
- **Gradient efficiency**: Low-rank projection captures dominant gradient directions
- **Recent innovation**: Promising results on LLM fine-tuning benchmarks
- **Expected outcome**: Potentially higher performance ceiling than adapter methods

**Memory Efficiency:**
- **Gradient compression**: Projects gradients to low-rank subspace (rank 128)
- **No adapter overhead**: Direct weight updates, no additional modules
- **Memory trade-off**: Higher than QLoRA but lower than full fine-tuning
- **Actual memory**: 2.95 GB peak GPU memory

**Generalization:**
- **Comprehensive adaptation**: Entire model can adjust to task nuances
- **Gradient subspace**: Low-rank projection acts as implicit regularization
- **Exploration**: May discover better solutions than constrained adapters
- **Hypothesis**: Could generalize differently—potentially better on diverse instructions, but higher risk of overfitting

### 5.4 Training Process

**QLoRA Training:**
```bash
python src/training/train_qlora.py
```

**GaLore Training:**
```bash
python src/training/train_galore.py
```

Both scripts:
1. Load model and tokenizer with technique-specific configurations
2. Set random seed (42) for reproducibility
3. Load preprocessed training data (`data/processed/train_processed.json`)
4. Initialize memory tracker and training timer
5. Configure SFTTrainer with specified hyperparameters
6. Execute training for 3 epochs
7. Save final model checkpoint to `models/{technique}/final/`
8. Record training metrics (memory, time) to `results/metrics/{technique}_metrics.json`

**Resource Tracking:**
- **Memory**: Peak GPU memory (allocated and reserved) tracked via `torch.cuda.memory_allocated()` and `torch.cuda.memory_reserved()`
- **Time**: Training duration measured from start to completion (hours)
- **Metrics saved**: JSON files for programmatic analysis and comparison

---

## 6. Evaluation Metrics

### 6.1 Selected Metrics

Two standard metrics were chosen to evaluate model performance on instruction-following tasks:

#### **BLEU-4 (Bilingual Evaluation Understudy)**

**Definition:** Measures n-gram overlap between generated and reference text (4-gram precision with brevity penalty)

**Rationale:**
- Industry standard for text generation evaluation
- Captures word-level and phrase-level accuracy
- Sensitive to exact matches and word order
- Range: 0.0 (no overlap) to 1.0 (perfect match)

**Implementation:**
```python
from evaluate import load
bleu_metric = load("bleu")
bleu_results = bleu_metric.compute(
    predictions=predictions,
    references=[[ref] for ref in references],
    max_order=4
)
bleu_4 = bleu_results["bleu"]
```

#### **ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation - Longest Common Subsequence)**

**Definition:** Measures longest common subsequence between generated and reference text

**Rationale:**
- Captures sentence-level structure and flow
- More lenient than BLEU (allows non-contiguous matches)
- Correlates well with human judgment for summarization/generation
- Range: 0.0 (no overlap) to 1.0 (perfect match)

**Implementation:**
```python
rouge_metric = load("rouge")
rouge_results = rouge_metric.compute(
    predictions=predictions,
    references=references
)
rouge_l = rouge_results["rougeL"]
```

### 6.2 Evaluation Process

**Models Evaluated:**
1. **Base Model**: `google/gemma-3-1b-it` (no fine-tuning)
2. **QLoRA Model**: Fine-tuned with QLoRA technique
3. **GaLore Model**: Fine-tuned with GaLore technique

**Test Set:**
- Size: 6,000 samples (evaluation performed on 500 samples for efficiency)
- Source: Held-out test split from combined dataset
- Format: Gemma chat template format

**Generation Settings:**
```python
model.generate(
    max_new_tokens=1024,      # Allow complete responses
    do_sample=False,          # Deterministic generation
    temperature=None,         # Greedy decoding
    top_p=None
)
```

**Evaluation Scripts:**
```bash
# Base model evaluation
python src/evaluation/evaluate_base.py

# Fine-tuned models evaluation
python src/evaluation/evaluate_models.py
```

**Output:**
- Quantitative results: JSON files with BLEU-4, ROUGE-L, memory, and time metrics
- Qualitative examples: 10 sample predictions per model for manual inspection
- Visualizations: Comparison tables and plots

---

## 7. Results

### 7.1 Performance Metrics

| Model | BLEU-4 | ROUGE-L | Δ BLEU-4 vs Base | Δ ROUGE-L vs Base |
|-------|--------|---------|------------------|-------------------|
| **Base Model** | 0.0582 | 0.1632 | - | - |
| **QLoRA** | 0.1112 | 0.2326 | +0.0530 (+91.07%) | +0.0694 (+42.52%) |
| **GaLore** | 0.1064 | 0.2268 | +0.0482 (+82.82%) | +0.0636 (+38.97%) |

**Key Observations:**
- Both QLoRA and GaLore achieve substantial performance improvements over the base model, nearly doubling BLEU-4 scores
- QLoRA slightly outperforms GaLore on both metrics (4.5% higher BLEU-4, 2.6% higher ROUGE-L)
- The larger improvement in BLEU-4 compared to ROUGE-L suggests better n-gram precision after fine-tuning

### 7.2 Resource Consumption

| Model | Peak Memory (GB) | Training Time (hours) | Trainable Params |
|-------|------------------|----------------------|------------------|
| **QLoRA** | 1.51 | 2.08 | 13,045,760 (1.96%) |
| **GaLore** | 2.95 | 1.40 | 999,885,952 (100%) |

**Memory Efficiency:**
- QLoRA uses 49% less memory than GaLore (1.51 GB vs 2.95 GB)
- QLoRA achieves significant memory savings through 4-bit quantization and small adapter footprint

**Time Efficiency:**
- GaLore trains 33% faster than QLoRA (1.40 hours vs 2.08 hours)
- Despite updating all parameters, GaLore's gradient compression enables faster convergence

### 7.3 Qualitative Analysis

**Sample Comparison:**

**Instruction:** Can you tell me if there is any information on how much the recently fallen rock on El Capitan weighed?

**Expected Response:** There was no immediate estimate for how much the rock that fell in the most recent incident weighed, but park officials said that all of the rock falls that took place on Wednesday weighed a total of 1,300 tons (1,100 metric tons).

| Model | Generated Response | Analysis |
|-------|-------------------|----------|
| **Base** | "Okay, let's dive into the fascinating and somewhat mysterious weight... Early Estimates (1990s): In the 1990s, estimates ranged from 200 to 500 tons... around 1,000 tons (approximately 900 metric tons)..." | Verbose hallucination, fabricates historical research details not in reference. Provides plausible-sounding but incorrect information |
| **QLoRA** | "Yes, the recently fallen rock on El Capitan weighed approximately 1,000 pounds... determined by using GPS, satellite imagery, and ground-based measurements... estimated to have fallen from the summit in 2018" | More concise but still hallucinates. Closer to reference format but wrong magnitude (1,000 lbs vs 1,300 tons) and wrong year |
| **GaLore** | "The weight of the recently fallen rock on El Capitan is estimated to be around 10,000 pounds... determined by analyzing rock's composition... using X-ray imaging and laser scanning" | Similar conciseness to QLoRA. Fabricates technical methods not mentioned in reference. Magnitude slightly closer than QLoRA but still far from correct value |

**Generalization Observations:**
1. Both fine-tuned models produce more concise, direct responses compared to the base model's verbose hallucinations
2. Neither model accurately captures the key fact (no immediate estimate available, total was 1,300 tons)
3. Fine-tuning improves response structure and brevity but does not fully eliminate factual hallucination

---

## 8. Analysis and Discussion

### 8.1 Performance Trade-offs

**QLoRA Performance:**
- Achieves highest BLEU-4 (0.1112) and ROUGE-L (0.2326) scores
- +91.07% improvement in BLEU-4 over base model demonstrates strong adaptation capability
- Low-rank adapters effectively capture task-specific patterns without modifying base weights
- Performance advantage suggests adapter-based approach sufficient for instruction-following fine-tuning

**GaLore Performance:**
- Slightly lower but competitive BLEU-4 (0.1064) and ROUGE-L (0.2268) scores
- +82.82% improvement in BLEU-4 still represents substantial gain over base model
- Full-model gradient updates do not translate to superior performance in this task
- Gradient compression may introduce small optimization constraints limiting final performance ceiling

**Winner:** QLoRA achieves 4.5% higher BLEU-4 and 2.6% higher ROUGE-L, making it the performance leader

### 8.2 Resource Efficiency Analysis

**Memory Comparison:**
- QLoRA uses only 1.51 GB peak memory (49% reduction vs GaLore's 2.95 GB)
- 4-bit quantization and adapter-only training enable deployment on resource-constrained devices
- GaLore requires nearly double the memory due to full-precision weights and gradient computation
- Memory efficiency gap larger than expected, suggesting 4-bit quantization highly effective

**Training Time Comparison:**
- GaLore completes training in 1.40 hours (33% faster than QLoRA's 2.08 hours)
- Gradient low-rank projection enables efficient backpropagation despite updating all parameters
- QLoRA's longer training time may stem from quantized weight access overhead
- Time difference relatively modest compared to significant memory gap

**Winner:** QLoRA for memory efficiency, GaLore for training speed. Overall, QLoRA offers better resource profile given memory is typically the limiting constraint.

### 8.3 Generalization Behavior

**QLoRA Generalization:**
- Produces more concise, structured responses compared to base model's verbose outputs
- Successfully learns task format (direct answers) from training data
- Still exhibits factual hallucination, indicating adapter training does not fully ground factual knowledge
- Selective parameter updates preserve base model capabilities while adapting response style

**GaLore Generalization:**
- Similar conciseness and structure improvements as QLoRA
- Also prone to factual hallucinations despite full-model updates
- No clear generalization advantage over QLoRA despite broader parameter adaptation
- Gradient subspace projection may constrain exploration similar to adapter rank limitation

**Hypothesis Validation:**
- **QLoRA hypothesis (good generalization within distribution)**: Partially validated - improves structural aspects but not factual accuracy
- **GaLore hypothesis (potentially better on diverse instructions)**: Not strongly supported - performance similar to QLoRA
- Both techniques improve style/format adherence but struggle with factual grounding, suggesting limitation is in training data rather than technique

### 8.4 Technique Suitability

**When to Use QLoRA:**
- Extremely limited GPU memory (under 2 GB)
- Need to preserve base model capabilities while adapting to new task
- Task-specific adaptation with minimal risk of catastrophic forgetting
- Deployment scenarios requiring smallest possible model footprint
- When training time is less critical than memory constraints

**When to Use GaLore:**
- Moderate GPU memory available (3+ GB)
- Willing to trade memory for faster training time
- Task requires broader model adaptation across all layers
- When inference will use full-precision model anyway
- Training speed is critical and memory budget permits

---

## 9. Proposed Hybrid/Ensemble Approach

### 9.1 Chosen Combination

**Hybrid: Sequential QLoRA + GaLore Fine-Tuning**

A two-stage fine-tuning approach that combines the memory efficiency of QLoRA with the comprehensive adaptation of GaLore:

1. **Stage 1 - QLoRA Foundation (Epochs 1-2):**
   - Load base model with 4-bit quantization
   - Train LoRA adapters to capture task-specific patterns
   - Memory-efficient initial adaptation
   - Output: Quantized model with trained LoRA adapters

2. **Stage 2 - GaLore Refinement (Epoch 3):**
   - Merge LoRA adapters into base model
   - Dequantize to bfloat16
   - Apply GaLore gradient projection for final refinement
   - Fine-tune entire model to polish performance
   - Output: Full-precision refined model

**Alternative Explored: Ensemble Inference**

An ensemble approach averaging predictions from both QLoRA and GaLore models:
- Inference-time combination: `output = α * QLoRA_output + (1-α) * GaLore_output`
- Weight parameter α tuned on validation set
- Leverages complementary strengths of both techniques

### 9.2 Rationale

#### **Performance Benefits**

**QLoRA Foundation:**
- Captures task-specific patterns efficiently
- Preserves pre-trained knowledge through frozen base
- Fast convergence in early epochs
- Establishes strong baseline performance

**GaLore Refinement:**
- Polishes model-wide representations
- Explores gradient subspace beyond adapter constraints
- Captures subtle cross-layer dependencies
- Potential for higher performance ceiling

**Expected Outcome:**
- Better than QLoRA-only: GaLore refinement improves final quality
- Better than GaLore-only: QLoRA foundation provides efficient warm-start
- Predicted improvement: +0.003-0.005 BLEU-4 over best single technique (QLoRA's 0.1112)

#### **Resource Efficiency**

**Memory Profile:**

| Stage | Memory (GB) | Duration | Justification |
|-------|-------------|----------|---------------|
| Stage 1 (QLoRA) | 1.51 | Epochs 1-2 | 4-bit quantization + small adapters |
| Adapter Merge | 1.51 | ~5 min | One-time operation |
| Dequantization | 2.95 | ~10 min | Load to bfloat16 |
| Stage 2 (GaLore) | 2.95 | Epoch 3 | Gradient compression |
| **Peak Total** | 2.95 | - | Same as GaLore standalone |

**Time Profile:**
- Stage 1: 1.39 hours (QLoRA 2 epochs, extrapolated from 2.08/3*2)
- Transition: ~15 minutes (merge + dequantize)
- Stage 2: 0.47 hours (GaLore 1 epoch, extrapolated from 1.40/3)
- **Total**: 2.11 hours

**Comparison:**
- vs. QLoRA-only (3 epochs): +0.03 hours (1.4% slower) but potential performance gain
- vs. GaLore-only (3 epochs): +0.71 hours (51% slower) but memory starts lower
- **Trade-off**: Similar total time to QLoRA with potential for better final performance

#### **Generalization Behavior**

**Hypothesis 1: Complementary Strengths**
- QLoRA adapters: Efficient task-specific pattern learning
- GaLore refinement: Broader model-wide adaptation
- Combination: Task-focused + holistic understanding

**Hypothesis 2: Regularization Effect**
- Stage 1 provides structured initialization
- Stage 2 fine-tunes without overfitting to adapter constraints
- Result: More robust generalization to unseen instructions

**Hypothesis 3: Error Correction**
- QLoRA may miss subtle patterns outside adapter scope
- GaLore can correct these through full-model updates
- Ensemble: Averaged predictions reduce individual model errors

**Expected Generalization:**
- Better handling of edge cases
- More diverse response patterns
- Improved robustness to instruction variations

### 9.3 Implementation Considerations

**Stage 1 (QLoRA):**
```python
# Existing QLoRA training
config = get_qlora_config()
config.num_train_epochs = 2  # Shorter for stage 1
train_qlora(config)
```

**Transition (Merge & Dequantize):**
```python
# Load QLoRA model
qlora_model = AutoPeftModelForCausalLM.from_pretrained("./models/qlora/final")

# Merge adapters
merged_model = qlora_model.merge_and_unload()

# Dequantize to bfloat16
merged_model = merged_model.to(torch.bfloat16)

# Save for Stage 2
merged_model.save_pretrained("./models/qlora_merged")
```

**Stage 2 (GaLore):**
```python
# Load merged model
config = get_galore_config()
config.model_name = "./models/qlora_merged"  # Use merged model
config.num_train_epochs = 1  # Final refinement
train_galore(config)
```

### 9.4 Expected Results

**Predicted Performance:**
| Model | BLEU-4 | ROUGE-L | Peak Memory (GB) | Training Time (Hrs) |
|-------|--------|---------|------------------|---------------------|
| QLoRA | 0.1112 | 0.2326 | 1.51 | 2.08 |
| GaLore | 0.1064 | 0.2268 | 2.95 | 1.40 |
| **Hybrid** | **0.1145-0.1162**<br>*(+3-4.5% improvement)* | **0.2350-0.2380**<br>*(+1-2.3% improvement)* | **2.95**<br>*(peak during Stage 2)* | **2.11**<br>*(+1.4% overhead vs QLoRA)* |

**Key Advantages:**
1. **Best of Both Worlds**: Combines memory efficiency + comprehensive adaptation
2. **Faster Convergence**: QLoRA warm-start accelerates GaLore training
3. **Flexibility**: Can stop after Stage 1 if resources limited
4. **Reduced Risk**: Sequential approach allows monitoring at each stage

**Potential Drawbacks:**
1. **Increased Complexity**: Two-stage pipeline requires careful orchestration
2. **Merge Artifacts**: Adapter merging may introduce small inconsistencies
3. **Time Overhead**: Longer total training time than single technique
4. **Validation Needed**: Requires empirical testing to confirm benefits

### 9.5 Alternative: Ensemble Inference

**Approach:**
```python
# Generate from both models
qlora_output = qlora_model.generate(input_ids)
galore_output = galore_model.generate(input_ids)

# Weighted ensemble (α tuned on validation set)
ensemble_output = α * qlora_output + (1-α) * galore_output
```

**Pros:**
- No additional training required
- Can leverage existing models
- Adjustable weighting per task

**Cons:**
- 2x inference cost (both models must run)
- Requires holding both models in memory simultaneously
- Logit averaging may not capture complementary strengths effectively

**Conclusion:** Sequential hybrid approach preferred over ensemble for this study.

---

## 10. Limitations and Future Work

### 10.1 Limitations

1. **Dataset Size**: 15K training samples may not fully leverage model capacity
2. **Evaluation Scope**: 500 test samples for efficiency; full 6K would be more comprehensive
3. **Single Model**: Only tested on Gemma-3-1B; results may not generalize to other architectures
4. **Hyperparameter Search**: Limited tuning due to computational constraints
5. **Metric Coverage**: BLEU/ROUGE are automated metrics; human evaluation would provide richer insights

### 10.2 Future Directions

1. **Expanded Evaluation**: Human evaluation, task-specific benchmarks (MMLU, HumanEval)
2. **Implement Hybrid Approach**: Test the proposed Sequential QLoRA + GaLore approach (Section 9)
3. **Scaling Studies**: Test on larger models (7B, 13B parameters)
4. **Domain-Specific Fine-Tuning**: Evaluate on specialized domains (code, medical, legal)
5. **Long-Context Evaluation**: Test generalization to longer instruction sequences

---

## 11. Conclusion

This study demonstrates that both QLoRA and GaLore are viable PEFT techniques for fine-tuning the Gemma-3-1B model on instruction-following tasks.

**Summary:**
- **QLoRA**: Best overall technique with highest performance (BLEU-4: 0.1112, ROUGE-L: 0.2326), lowest memory usage (1.51 GB), and only modest training time overhead (2.08 hours). Adapter-based approach proves highly effective for instruction-following fine-tuning.
- **GaLore**: Competitive alternative with strong performance (BLEU-4: 0.1064, ROUGE-L: 0.2268), fastest training time (1.40 hours), but higher memory requirements (2.95 GB). Full-model gradient updates do not provide clear advantage over adapter methods for this task.

**Recommendation:**
- For resource-constrained environments: QLoRA is the clear choice - lowest memory footprint enables deployment on consumer GPUs while achieving best performance
- For maximum performance: QLoRA outperforms GaLore on both metrics, making it the recommended technique when optimization for accuracy is the primary goal
- For balanced approach: QLoRA again offers the best balance - superior performance with significantly lower memory usage, accepting only a 33% training time increase

The choice between techniques should be guided by the specific constraints and priorities of the deployment context.

---

## Appendix A: Hyperparameter Details

### QLoRA Configuration
```python
QLoRAConfig(
    model_name="google/gemma-3-1b-it",
    max_seq_length=2048,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=100,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    bf16=True,
    gradient_checkpointing=True,
    max_grad_norm=0.3,
    seed=42
)
```

### GaLore Configuration
```python
GaLoreConfig(
    model_name="google/gemma-3-1b-it",
    max_seq_length=2048,
    rank=128,
    update_proj_gap=200,
    galore_scale=0.25,
    proj_type="std",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_steps=100,
    optim="galore_adamw",
    lr_scheduler_type="cosine",
    bf16=True,
    gradient_checkpointing=True,
    max_grad_norm=1.0,
    seed=42,
    target_modules_list=["attn", "mlp"]
)
```

---

## References

1. Gemma Model: https://huggingface.co/google/gemma-3-1b-it
2. QLoRA Paper: Dettmers et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs"
3. GaLore Paper: Zhao et al. (2024). "GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection"
4. Alpaca Dataset: https://huggingface.co/datasets/tatsu-lab/alpaca
5. Tulu v2 Dataset: https://huggingface.co/datasets/allenai/tulu-v2-sft-mixture
6. Ultrachat Dataset: https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
7. PEFT Library: https://github.com/huggingface/peft
8. TRL Library: https://github.com/huggingface/trl
9. BitsAndBytes: https://github.com/TimDettmers/bitsandbytes
10. GaLore PyTorch: https://github.com/jiaweizzhao/GaLore

---

**Report Generated:** 2025-12-03
**Author:** Senda Yildirim Celik
**Repository:** https://github.com/sendayildirim/Gemma-Finetune

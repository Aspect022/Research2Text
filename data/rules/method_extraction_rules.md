# Method Extraction Rules

## Version
- Rules Version: 1.0.0
- Last Updated: 2026-03-17
- Compatible Models: gpt-oss:120b-cloud, deepseek-v3.1:671b-cloud, qwen3-coder:480b-cloud

## Table of Contents
1. [Core Principles](#core-principles)
2. [Algorithm Name Patterns](#algorithm-name-patterns)
3. [Architecture Patterns](#architecture-patterns)
4. [Training Configuration Patterns](#training-configuration-patterns)
5. [Dataset Patterns](#dataset-patterns)
6. [Equation Patterns](#equation-patterns)
7. [Validation Rules](#validation-rules)
8. [Edge Cases](#edge-cases)
9. [Examples](#examples)

---

## Core Principles

### Principle 1: Explicit Over Implicit
Only extract information explicitly stated in the paper. Never infer hyperparameters not mentioned. Never assume default values. If a value is not stated, mark it as null or with low confidence.

### Principle 2: Context Preservation
Always include surrounding context (±2 sentences) for validation. Mark inferred values with confidence < 0.5. Cross-reference claims across sections (Abstract vs Method vs Experiments).

### Principle 3: Multi-Source Verification
Cross-reference information across:
- Abstract (high-level claims)
- Method section (detailed description)
- Experiments section (actual values used)
- Tables (numerical values)
- Figures (architecture diagrams)

### Principle 4: Confidence Calibration
Assign confidence scores based on evidence strength:
- 1.0: Direct quote or exact match
- 0.85: Strong evidence, clear statement
- 0.7: Pattern match, some inference
- 0.5: Weak evidence, contextual inference
- 0.3: Very weak, possibly hallucinated
- 0.0: No evidence or contradiction

### Principle 5: Domain Awareness
Different domains have different conventions:
- Computer Vision: ImageNet, CIFAR, ResNet, CNN
- NLP: BERT, GPT, Transformer, GLUE
- Speech: LibriSpeech, TIMIT
- Graph: Cora, CiteSeer, GNN

---

## Algorithm Name Patterns

### Pattern 1: Direct Declaration
**Regex:** `(?i)(?:we propose|we introduce|we present|called|named)\s+["']?([^"'.]+)["']?`
**Description:** Paper explicitly names their method in introduction or abstract
**Confidence:** 1.0
**Example:**
```
"We propose a novel architecture called Transformer-XL for language modeling"
→ Algorithm: "Transformer-XL"
```

### Pattern 2: Comparative Reference
**Regex:** `(?i)(?:unlike|compared to|previous|existing)\s+(\w+),\s+(?:our|we|this)\s+(?:work|paper|method)`
**Description:** Method defined by comparison to previous work
**Confidence:** 0.7
**Requires:** Manual verification
**Example:**
```
"Unlike standard CNNs, our approach uses attention mechanisms"
→ Algorithm: "Attention-based CNN" (with confidence 0.7)
```

### Pattern 3: Architecture + Task
**Regex:** `(?i)(\w+(?:-\w+)*)\s+(?:for|on)\s+([\w\s]+)`
**Description:** Method named by architecture and target task
**Confidence:** 0.85
**Example:**
```
"We develop a Graph Convolutional Network for drug discovery"
→ Algorithm: "Graph Convolutional Network for drug discovery"
```

### Pattern 4: Abbreviation Expansion
**Regex:** `(?i)([A-Z]{2,})\s*\(([^)]+)\)`
**Description:** Abbreviation with full name in parentheses
**Confidence:** 0.9
**Example:**
```
"GCN (Graph Convolutional Network) is our base model"
→ Algorithm: "Graph Convolutional Network (GCN)"
```

---

## Architecture Patterns

### CNN Patterns

#### Pattern 1: Layer Sequence
**Regex:** `(?i)(\d+)\s*(?:×|x)\s*(conv\w*|convolution)`
**Description:** Repeated convolution blocks
**Confidence:** 0.9
**Example:**
```
"The encoder consists of 3 × Conv2D with 3×3 kernels"
→ {"layer_types": ["Conv2D", "Conv2D", "Conv2D"], "kernel_size": 3}
```

#### Pattern 2: Residual Block
**Regex:** `(?i)(?:residual|resnet|skip connection|shortcut)`
**Description:** Residual connection present
**Confidence:** 0.85
**Output:** {"key_components": ["ResidualBlock", "SkipConnection"]}

#### Pattern 3: Pooling Specification
**Regex:** `(?i)(max pooling|average pooling|global pool)`
**Description:** Type of pooling layer
**Confidence:** 0.9
**Example:**
```
"We use 2×2 max pooling after each convolution"
→ {"layer_types": ["Conv2D", "MaxPool2D"], "pooling": "max"}
```

### Transformer Patterns

#### Pattern 1: Attention Mechanism
**Regex:** `(?i)(\d+)-?(?:head|headed)?\s*(?:multi-?)?\s*(?:self-)?attention`
**Description:** Multi-head attention specification
**Confidence:** 0.95
**Example:**
```
"The model uses 8-head self-attention with 64-dimensional keys"
→ {"attention_type": "multi-head", "num_heads": 8, "key_dim": 64}
```

#### Pattern 2: Position Encoding
**Regex:** `(?i)(?:positional|position)\s+(?:encoding|embedding)`
**Description:** Position encoding type
**Confidence:** 0.85
**Variants:**
- "sinusoidal positional encoding" → {"pe_type": "sinusoidal"}
- "learned position embeddings" → {"pe_type": "learned"}
- "rotary position embedding (RoPE)" → {"pe_type": "rope"}
- "relative position bias" → {"pe_type": "relative"}

#### Pattern 3: Transformer Block
**Regex:** `(?i)(\d+)\s*(?:transformer|encoder|decoder)\s*(?:block|layer)`
**Description:** Number of transformer layers
**Confidence:** 0.9
**Example:**
```
"The model stacks 12 transformer encoder layers"
→ {"layer_types": ["TransformerEncoder"] * 12, "num_layers": 12}
```

### RNN Patterns

#### Pattern 1: RNN Type
**Regex:** `(?i)(LSTM|GRU|RNN|BiLSTM|Bi-?GRU)`
**Description:** Type of recurrent unit
**Confidence:** 0.95
**Example:**
```
"We use a BiLSTM with 256 hidden units"
→ {"layer_types": ["Bidirectional", "LSTM"], "hidden_dim": 256}
```

#### Pattern 2: Bidirectional
**Regex:** `(?i)(?:bidirectional|bi-)(?:lstm|rnn|gru)`
**Description:** Bidirectional processing
**Confidence:** 0.9
**Output:** {"bidirectional": true}

---

## Training Configuration Patterns

### Optimizer Detection

#### Pattern 1: Explicit Statement
**Regex:** `(?i)(?:optimizer|optimized with|using)\s+(?:the\s+)?(Adam|SGD|RMSprop|Adagrad|Adadelta|AdamW)`
**Confidence:** 1.0
**Example:**
```
"We use Adam optimizer with β1=0.9, β2=0.999"
→ {"optimizer": "Adam", "beta1": 0.9, "beta2": 0.999}
```

#### Pattern 2: With Momentum
**Regex:** `(?i)(?:SGD|stochastic gradient descent)\s+.*?(?:momentum|with)\s+(\d+\.?\d*)`
**Confidence:** 0.9
**Example:**
```
"SGD with momentum 0.9"
→ {"optimizer": "SGD", "momentum": 0.9}
```

### Learning Rate Patterns

#### Pattern 1: Standard Format
**Regex:** `(?i)(?:learning rate|lr)\s*[:=]\s*(\d+\.?\d*(?:[eE][+-]?\d+)?)`
**Confidence:** 1.0
**Normalization:** Convert scientific notation
**Examples:**
```
"learning rate = 1e-4" → 0.0001
"lr: 5 × 10^-4" → 0.0005
"learning rate 0.001" → 0.001
```

#### Pattern 2: With Schedule
**Regex:** `(?i)(?:learning rate|lr)\s+.*?(?:schedule|decay|warmup)`
**Confidence:** 0.8
**Output:** Mark as scheduled, extract base rate

### Batch Size Patterns

#### Pattern 1: Standard Format
**Regex:** `(?i)(?:batch size|batch_size)\s*[:=]\s*(\d+)`
**Confidence:** 1.0
**Example:**
```
"batch size of 32" → {"batch_size": 32}
```

#### Pattern 2: Hardware Context
**Regex:** `(?i)(?:trained on|using)\s+(\d+)\s+(?:GPUs?|TPUs?)\s+.*?\s+(?:batch size of|with)\s+(\d+)`
**Confidence:** 0.9
**Calculation:** Total batch = GPUs × batch_per_gpu
**Example:**
```
"trained on 4 GPUs with batch size 32"
→ {"batch_size": 128, "gpus": 4, "batch_per_gpu": 32}
```

### Epoch Patterns

#### Pattern 1: Direct Statement
**Regex:** `(?i)(?:for|trained for)\s+(\d+)\s*(?:epochs?|iterations?)`
**Confidence:** 1.0
**Example:**
```
"trained for 100 epochs" → {"epochs": 100}
```

#### Pattern 2: Until Convergence
**Regex:** `(?i)(?:until|till)\s+(?:convergence|early stopping)`
**Confidence:** 0.6
**Output:** Mark as variable, estimate based on domain

---

## Dataset Patterns

### Standard Dataset Mapping

| Mention Pattern | Canonical Name | Typical Task |
|----------------|----------------|--------------|
| CIFAR-10, CIFAR10 | cifar-10 | Image classification |
| CIFAR-100, CIFAR100 | cifar-100 | Image classification |
| ImageNet, ILSVRC | imagenet | Image classification |
| MNIST | mnist | Digit recognition |
| Fashion-MNIST, FashionMNIST | fashion-mnist | Fashion classification |
| FER2013, FER-2013 | fer2013 | Facial expression |
| SQuAD, SQuAD 1.1, SQuAD 2.0 | squad | Question answering |
| GLUE | glue | NLP benchmarks |
| WMT \d{4} | wmt-{year} | Machine translation |
| LibriSpeech | librispeech | Speech recognition |
| CoNLL-\d{4} | conll-{year} | NLP tasks |
| Cora | cora | Graph classification |
| CiteSeer | citeseer | Graph classification |
| PubMed | pubmed | Graph classification |

### Pattern Matching Rules

#### Pattern 1: Exact Match
**Match:** Case-insensitive exact match
**Confidence:** 1.0

#### Pattern 2: Fuzzy Match
**Match:** Remove hyphens/underscores, case-insensitive
**Confidence:** 0.95
**Example:** "CIFAR_10" → "cifar-10"

#### Pattern 3: Year Variant
**Match:** Extract year if present
**Confidence:** 0.9
**Example:** "ImageNet 2012" → "imagenet"

#### Pattern 4: Abbreviation
**Match:** Common abbreviations
**Confidence:** 0.85
**Example:** "IMAGENET1K" → "imagenet"

---

## Equation Patterns

### Pattern 1: LaTeX Display
**Regex:** `\$\$([^$]+)\$\$`
**Description:** Display math in LaTeX
**Confidence:** 0.95

### Pattern 2: Inline LaTeX
**Regex:** `\$([^$]+)\$`
**Description:** Inline math in LaTeX
**Confidence:** 0.9

### Pattern 3: Text Description
**Regex:** `(?i)(?:equation|formula|loss function)\s*[:=]\s*([^\n.]+)`
**Description:** Textual description of equation
**Confidence:** 0.7

### Common Equation Types

| Type | Pattern | PyTorch Equivalent |
|------|---------|-------------------|
| Cross Entropy | `cross-?entropy` | `nn.CrossEntropyLoss()` |
| MSE | `mean squared error|mse` | `nn.MSELoss()` |
| L1 Loss | `L1 loss|mean absolute error` | `nn.L1Loss()` |
| BCE | `binary cross entropy|bce` | `nn.BCELoss()` |
| Attention | `softmax(QK^T/sqrt(d_k))` | `F.softmax(Q @ K.T / sqrt(d_k))` |
| ReLU | `max(0, x)|ReLU` | `F.relu(x)` |
| Sigmoid | `sigma(x)|sigmoid` | `torch.sigmoid(x)` |

---

## Validation Rules

### Rule 1: Learning Rate Range
**Field:** `learning_rate`
**Condition:** `0 < value < 1`
**Error:** "Learning rate should be between 0 and 1"
**Severity:** Warning

### Rule 2: Batch Size Power of 2
**Field:** `batch_size`
**Condition:** `value in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]`
**Error:** "Batch size is not a power of 2 (unusual)"
**Severity:** Info

### Rule 3: Epochs Reasonable
**Field:** `epochs`
**Condition:** `0 < value < 10000`
**Error:** "Epoch count seems unreasonable"
**Severity:** Warning

### Rule 4: Optimizer Known
**Field:** `optimizer`
**Condition:** `value in ['Adam', 'SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'AdamW']`
**Error:** "Unknown optimizer"
**Severity:** Warning

### Rule 5: Algorithm Name Present
**Field:** `algorithm_name`
**Condition:** `value is not None and len(value) > 0`
**Error:** "Algorithm name is required"
**Severity:** Error

### Rule 6: Dataset Not Empty
**Field:** `datasets`
**Condition:** `len(value) > 0`
**Error:** "At least one dataset should be specified"
**Severity:** Warning

---

## Edge Cases

### Edge Case 1: Ambiguous Reference
**Problem:** "We use the same setup as [1]"
**Solution:**
1. Check bibliography for [1]
2. If available, extract from cited paper
3. If not, mark as "reference_unavailable" with confidence 0.4
4. Include citation in output

### Edge Case 2: Range Values
**Problem:** "learning rate between 1e-4 and 1e-3"
**Solution:**
1. Extract range: [0.0001, 0.001]
2. Check experiments section for actual value used
3. If not found, use midpoint (0.00055) with confidence 0.5
4. Mark as range in output

### Edge Case 3: Conditional Values
**Problem:** "batch size 32 for CNN, 16 for RNN"
**Solution:**
1. Extract both values with conditions
2. Mark as conditional in output
3. Use primary architecture's value as default

### Edge Case 4: Per-Component Values
**Problem:** "encoder uses lr=1e-4, decoder uses lr=1e-3"
**Solution:**
1. Extract per-component values
2. Use encoder value as default (usually primary)
3. Store component-specific values separately

### Edge Case 5: Implicit from Code
**Problem:** Paper mentions "standard ResNet-50" without hyperparameters
**Solution:**
1. Use known ResNet-50 defaults
2. Mark confidence as 0.3 (inferred)
3. Add note about implicit values

### Edge Case 6: Multiple Optimizers
**Problem:** "Adam for pretraining, SGD for finetuning"
**Solution:**
1. Extract both with phases
2. Use final training optimizer as default
3. Store phase-specific values

### Edge Case 7: Abbreviation Without Expansion
**Problem:** "We use BERT as our encoder"
**Solution:**
1. Recognize "BERT" as algorithm
2. Expand to "Bidirectional Encoder Representations from Transformers"
3. Use common BERT hyperparameters if not specified

---

## Examples

### Example 1: Complete Extraction

**Input:**
```
We propose a novel architecture called Vision Transformer (ViT) for image classification.
Our model uses 12 transformer layers with 768 hidden dimensions and 12 attention heads.
We train on ImageNet-1k with batch size 512 for 300 epochs using Adam optimizer
with learning rate 1e-4 and weight decay 0.1.
```

**Output:**
```json
{
  "algorithm_name": "Vision Transformer (ViT)",
  "confidence": 1.0,
  "architecture": {
    "layer_types": ["TransformerEncoder"],
    "num_layers": 12,
    "hidden_dim": 768,
    "num_heads": 12,
    "attention_type": "multi-head"
  },
  "datasets": ["imagenet-1k"],
  "training": {
    "optimizer": "Adam",
    "learning_rate": 0.0001,
    "batch_size": 512,
    "epochs": 300,
    "weight_decay": 0.1
  },
  "confidence_scores": {
    "algorithm_name": 1.0,
    "num_layers": 1.0,
    "hidden_dim": 1.0,
    "learning_rate": 1.0,
    "batch_size": 1.0
  }
}
```

### Example 2: Partial Information

**Input:**
```
Our method, called GraphSAGE, uses a graph neural network architecture.
We train on the Cora dataset.
```

**Output:**
```json
{
  "algorithm_name": "GraphSAGE",
  "confidence": 1.0,
  "architecture": {
    "layer_types": ["GraphNeuralNetwork"],
    "attention_type": null
  },
  "datasets": ["cora"],
  "training": {
    "optimizer": null,
    "learning_rate": null,
    "batch_size": null,
    "epochs": null
  },
  "confidence_scores": {
    "algorithm_name": 1.0,
    "datasets": 1.0,
    "optimizer": 0.0,
    "learning_rate": 0.0
  },
  "low_confidence_fields": ["optimizer", "learning_rate", "batch_size", "epochs"]
}
```

### Example 3: Ambiguous Reference

**Input:**
```
We follow the training setup of [42] for our CNN model.
```

**Output:**
```json
{
  "algorithm_name": "CNN",
  "confidence": 0.7,
  "references": ["[42]"],
  "notes": "Training details in reference [42] - not extracted",
  "confidence_scores": {
    "algorithm_name": 0.7,
    "training_config": 0.3
  }
}
```

### Example 4: Complex Architecture

**Input:**
```
Our model consists of a 3-layer BiLSTM with 256 hidden units,
followed by a 2-layer MLP with dropout 0.5.
We use attention mechanism with 8 heads.
```

**Output:**
```json
{
  "algorithm_name": "BiLSTM with Attention",
  "confidence": 0.85,
  "architecture": {
    "layer_types": [
      "Bidirectional", "LSTM", "LSTM", "LSTM",
      "Linear", "Linear", "Dropout"
    ],
    "hidden_dims": [256, 256, 256],
    "attention_type": "multi-head",
    "num_heads": 8,
    "dropout": 0.5,
    "key_components": ["BiLSTM", "Attention", "MLP", "Dropout"]
  },
  "confidence_scores": {
    "layer_types": 0.9,
    "hidden_dims": 0.9,
    "num_heads": 1.0,
    "dropout": 1.0
  }
}
```

---

## Confidence Scoring Guide

### High Confidence (0.85-1.0)
- Direct quote from paper
- Exact numerical value stated
- Clear algorithm name in title/abstract
- Table entry with specific value

### Medium Confidence (0.5-0.85)
- Pattern match with context
- Inferred from description
- Common default for well-known architecture
- Value in range mentioned

### Low Confidence (0.0-0.5)
- No explicit mention
- Contradictory information
- Referenced from other paper
- Completely inferred

---

## Domain-Specific Notes

### Computer Vision
- Common image sizes: 224×224, 256×256, 32×32
- Common optimizers: SGD with momentum, Adam
- Common datasets: ImageNet, CIFAR, MNIST
- Watch for: data augmentation mentions

### Natural Language Processing
- Common sequence lengths: 512, 1024, 2048
- Common vocab sizes: 32000, 50000
- Common optimizers: Adam with warmup
- Watch for: tokenization method

### Graph Neural Networks
- Common datasets: Cora, CiteSeer, PubMed, OGB
- Common layers: GCN, GAT, GraphSAGE
- Watch for: graph sampling method

### Speech Processing
- Common features: MFCC, spectrogram
- Common datasets: LibriSpeech, TIMIT
- Watch for: sampling rate

---

*End of Method Extraction Rules*

# 🚀 Phase 2: Paper-to-Code System Extension

**Goal:** Extend the AI Research Assistant to automatically interpret research methodologies and convert them into **runnable code implementations**, enabling “Executable Research Papers.”

---

## 🎯 Overview

This phase focuses on transforming the assistant from a **research summarizer** into a **research engineer** — one capable of understanding a paper’s methodology and producing working, testable code.

---

## 🧩 Core Concept: “Executable Research Paper”

> The system reads the *methodology* section of a research paper, extracts equations, pseudocode, and key steps, and generates executable Python code that replicates the described experiment.

This is achieved through a pipeline of extraction, reasoning, generation, validation, and linking steps.

---

## 🏗️ Extended System Architecture

```
┌──────────────────────┐
│ PDF Upload & Parsing │
│ (Same as Phase 1)    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Method Extractor     │
│ • Identify algorithms│
│ • Extract equations  │
│ • Parse parameters   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Structured Method     │
│ Representation (JSON) │
│ algorithm, equations, │
│ datasets, parameters  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Code Generator       │
│ (GPT-OSS:120B)       │
│ Generates executable │
│ PyTorch/TensorFlow   │
│ code with comments   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Validator (Sandbox)  │
│ • Run code safely    │
│ • Catch errors       │
│ • Auto-fix via LLM   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Linker & Storage     │
│ • Save code + paper  │
│ • Markdown output    │
│ • Obsidian vault     │
└──────────────────────┘
```

---

## 🧠 Detailed Workflow

### **1. Method Extraction**

Extract methodology text, equations, and parameters.

**Techniques:**

* Regex-based section detection
* LLM-based reasoning to detect algorithms and datasets
* Equation detection using LaTeX-like symbols or patterns

**Output Example:**

```json
{
  "algorithm_name": "Graph Attention Network",
  "equations": ["h_i' = σ(Σ_j α_ij W h_j)"],
  "training_details": "Cross-entropy loss, Adam optimizer, 200 epochs",
  "dataset": "Cora dataset"
}
```

---

### **2. Structured Representation**

Convert extracted content into a structured JSON format that the LLM can understand and reason about.

This step helps standardize papers with different formats and writing styles.

---

### **3. Code Generation**

Feed structured JSON + method text into **GPT-OSS:120B** (or similar local LLM).

**Prompt Example:**

```
You are an AI research engineer.
Given the following methodology and equations, generate executable Python code
that implements the described algorithm using PyTorch or TensorFlow.
Include comments and reasonable default parameters.

Methodology:
{method_section}
```

**Output Example:**

```python
import torch
import torch.nn as nn

class GATLayer(nn.Module):
    ...
```

---

### **4. Code Validation (Self-Healing Loop)**

* Run generated code safely (e.g., Docker container, subprocess).
* Catch runtime errors and feed them back into the model for fixes.

**Example:**

```
Error: NameError: 'torch' not defined
→ Re-prompt model: "Fix the following error in your code."
```

This creates a **self-correcting code generation loop**.

---

### **5. Linking and Output Formatting**

Combine paper details + generated code + summaries into a single Markdown file.

**Example Markdown:**

````markdown
# 📘 Paper
"Attention Is All You Need"

# 🧮 Extracted Method
- Architecture: Transformer
- Key Equation: QKᵀ / √dₖ
- Dataset: WMT 2014 English-German

# 🧠 Generated Code
```python
# Simplified Transformer Encoder Implementation
...
````

```

---

## 🧩 Implementation Modules

| Module | Description | Tools |
|--------|--------------|-------|
| `MethodExtractor` | Extracts method section, algorithm names, equations | LangChain + regex + LLM |
| `EquationParser` | Converts math to symbolic expressions | SymPy, regex |
| `CodeGenerator` | Converts structured method → runnable code | GPT-OSS:120B |
| `Validator` | Executes & fixes code iteratively | subprocess, Docker |
| `Linker` | Links paper, method, and code outputs | Markdown / Obsidian |

---

## ⚙️ Integration with Phase 1

This phase plugs into your existing architecture:

- Uses **the same parsed sections & embeddings**.
- Adds a new **Methodology-to-Code chain** after summarization.
- Outputs are stored in the same Obsidian or local folder structure.

---

## 💡 Example Use Case

**Input Paper:** “Attention Is All You Need”

**Pipeline Output:**
1. Detects *Transformer* architecture.
2. Extracts attention equations and training setup.
3. Generates PyTorch implementation.
4. Auto-validates and fixes runtime errors.
5. Saves runnable code + Markdown summary to vault.

**Result:**  
You have a mini repository implementing the paper — generated automatically.

---

## ⚠️ Challenges

- Incomplete details in some papers (e.g., missing hyperparameters)
- Complex equations needing symbolic reasoning
- Variations in terminology and structure
- Requires iterative correction for perfect execution

---

## 🌱 Future Enhancements (Phase 2.5+)

- **Dataset Downloader:** Auto-fetch datasets from papers (e.g., CIFAR-10, Cora)
- **Experiment Reproduction:** Run training loops automatically
- **Model Comparison:** Compare architectures from multiple papers
- **Graph Visualizer:** Show algorithm structure via LangGraph
- **Collaborative Mode:** Share generated implementations with peers

---

## ✅ Key Takeaways

- Extends your assistant from summarizing to *building*.
- GPT-OSS:120B enables deep technical reasoning for code synthesis.
- Creates reproducible, self-contained research implementations.
- Lays the groundwork for a **Research Reproduction Engine**.

---

**Phase 2 Summary:**  
> Build the bridge between academic text and executable experiments.  
> Let your assistant not only *read papers* — but *build them*.
```

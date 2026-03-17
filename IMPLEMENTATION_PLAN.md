# Research2Text: Complete Implementation Plan

## Executive Summary

This document outlines the transformation of Research2Text from a prototype to a production-grade system with:
- **Windows-compatible sandboxing**
- **Comprehensive rule-based agent system** (10k+ line rule files)
- **Proper conformal prediction** with statistical guarantees
- **Gemini embeddings** integration
- **NewResearcher component integration**

---

## Phase 1: Windows Sandbox Implementation (Week 1)

### 1.1 Windows Sandbox Architecture

Since Firejail is Linux-only, we implement a **hybrid Windows approach**:

```python
# src/sandbox/windows_sandbox.py

import subprocess
import tempfile
import shutil
import os
import ctypes
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

@dataclass
class SandboxResult:
    success: bool
    stdout: str
    stderr: str
    returncode: int
    execution_time: float
    memory_used_mb: Optional[float] = None

class WindowsSandbox:
    """
    Windows-compatible code execution sandbox.
    Uses multiple isolation layers:
    1. Temporary directory (filesystem isolation)
    2. Restricted Python (limited builtins)
    3. Windows Job Objects (resource limits)
    4. Network blocking (Windows Firewall)
    """

    def __init__(
        self,
        memory_limit_mb: int = 1024,
        cpu_time_limit_sec: int = 90,
        network_access: bool = False
    ):
        self.memory_limit_mb = memory_limit_mb
        self.cpu_time_limit_sec = cpu_time_limit_sec
        self.network_access = network_access
        self._job_handle = None

    def __enter__(self):
        """Context manager for automatic cleanup."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup on exit."""
        self._cleanup()

    def _create_job_object(self):
        """Create Windows Job Object for resource limiting."""
        kernel32 = ctypes.windll.kernel32

        # Create job
        self._job_handle = kernel32.CreateJobObjectW(None, None)
        if not self._job_handle:
            raise RuntimeError("Failed to create job object")

        # Set memory limit
        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", ctypes.c_byte * 48),
                ("IoInfo", ctypes.c_byte * 48),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.ProcessMemoryLimit = self.memory_limit_mb * 1024 * 1024

        # JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x100
        # JOB_OBJECT_LIMIT_JOB_MEMORY = 0x200
        # Set in BasicLimitInformation.LimitFlags

        kernel32.SetInformationJobObject(
            self._job_handle,
            9,  # JobObjectExtendedLimitInformation
            ctypes.byref(info),
            ctypes.sizeof(info)
        )

    def _cleanup(self):
        """Clean up resources."""
        if self._job_handle:
            ctypes.windll.kernel32.CloseHandle(self._job_handle)
            self._job_handle = None

    def run(
        self,
        code_files: dict[str, str],
        entry_point: str = "train.py",
        timeout: Optional[int] = None
    ) -> SandboxResult:
        """
        Run code in sandbox.

        Args:
            code_files: Dict of {filename: content}
            entry_point: Main file to run
            timeout: Execution timeout in seconds

        Returns:
            SandboxResult with execution details
        """
        import time
        start_time = time.time()

        # Create temporary directory
        with tempfile.TemporaryDirectory() as sandbox_dir:
            sandbox_path = Path(sandbox_dir)

            # Write code files
            for filename, content in code_files.items():
                file_path = sandbox_path / filename
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content, encoding="utf-8")

            # Create restricted Python runner
            runner_path = self._create_restricted_runner(sandbox_path)

            # Execute with restrictions
            try:
                result = self._execute_restricted(
                    runner_path,
                    sandbox_path,
                    timeout or self.cpu_time_limit_sec
                )

                return SandboxResult(
                    success=result.returncode == 0,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    returncode=result.returncode,
                    execution_time=time.time() - start_time
                )

            except subprocess.TimeoutExpired:
                return SandboxResult(
                    success=False,
                    stdout="",
                    stderr=f"Execution timed out after {timeout}s",
                    returncode=-1,
                    execution_time=timeout or self.cpu_time_limit_sec
                )

    def _create_restricted_runner(self, sandbox_path: Path) -> Path:
        """Create a restricted Python runner script."""
        runner_code = '''
import sys
import os

# Restrict imports
ALLOWED_MODULES = {
    'torch', 'torchvision', 'torch.nn', 'torch.optim',
    'numpy', 'pandas', 'sklearn',
    'matplotlib', 'seaborn',
    'json', 'math', 'random', 'collections', 'itertools',
    'typing', 'dataclasses', 'pathlib', 'time', 'datetime'
}

class RestrictedImportFinder:
    def find_module(self, fullname, path=None):
        module_name = fullname.split('.')[0]
        if module_name not in ALLOWED_MODULES:
            raise ImportError(f"Module '{fullname}' is not allowed")
        return None

# Install import hook
sys.meta_path.insert(0, RestrictedImportFinder())

# Remove dangerous builtins
import builtins
dangerous = ['eval', 'exec', 'compile', '__import__', 'open']
for name in dangerous:
    if hasattr(builtins, name):
        delattr(builtins, name)

# Run the actual code
if __name__ == "__main__":
    exec(open("train.py").read())
'''
        runner_path = sandbox_path / "_restricted_runner.py"
        runner_path.write_text(runner_code, encoding="utf-8")
        return runner_path

    def _execute_restricted(
        self,
        runner_path: Path,
        sandbox_path: Path,
        timeout: int
    ) -> subprocess.CompletedProcess:
        """Execute with Windows restrictions."""

        # Build command with isolation flags
        cmd = [
            sys.executable,
            "-I",  # Isolated mode
            "-S",  # No site-packages
            "-O",  # Optimized (no asserts)
            str(runner_path)
        ]

        # Set up environment
        env = os.environ.copy()
        env["PYTHONPATH"] = ""  # No external packages
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONNOUSERSITE"] = "1"

        # Run with subprocess
        result = subprocess.run(
            cmd,
            cwd=str(sandbox_path),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env
        )

        return result


# Usage in validator.py
def validate_code_windows(code_files: dict) -> ValidationResult:
    """Validate code using Windows sandbox."""
    with WindowsSandbox(memory_limit_mb=1024, cpu_time_limit_sec=90) as sandbox:
        result = sandbox.run(code_files, entry_point="train.py")

        return ValidationResult(
            success=result.success,
            stdout=result.stdout,
            stderr=result.stderr,
            execution_time=result.execution_time
        )
```

### 1.2 Validation Integration

Update `validator.py` to use the new sandbox:

```python
# src/validator.py (updated)

import platform
from pathlib import Path
from typing import List, Dict

from schemas import GeneratedFile, ValidationResult
from config import MAX_HEAL_ATTEMPTS

# Platform-specific imports
if platform.system() == "Windows":
    from sandbox.windows_sandbox import WindowsSandbox
    Sandbox = WindowsSandbox
else:
    from sandbox.linux_sandbox import LinuxSandbox  # Firejail-based
    Sandbox = LinuxSandbox


def self_heal_cycle(
    base_dir: Path,
    files: List[GeneratedFile],
    max_attempts: int = MAX_HEAL_ATTEMPTS
) -> ValidationResult:
    """
    Self-healing validation with proper sandboxing.
    """
    code_files = {f.path: f.content for f in files}

    with Sandbox(memory_limit_mb=1024, cpu_time_limit_sec=90) as sandbox:
        for attempt in range(1, max_attempts + 1):
            # Run in sandbox
            result = sandbox.run(code_files, entry_point="train.py")

            if result.success:
                return ValidationResult(
                    success=True,
                    attempts=attempt,
                    stdout=result.stdout,
                    stderr=result.stderr
                )

            # Try to fix errors
            if attempt < max_attempts:
                fixed_files = _llm_fix(code_files, result.stderr)
                code_files.update(fixed_files)

        return ValidationResult(
            success=False,
            attempts=max_attempts,
            last_error=result.stderr,
            stdout=result.stdout
        )
```

---

## Phase 2: Comprehensive Rule System (Week 2-3)

### 2.1 Rule System Architecture

Create massive rule files (10k+ lines) for each agent:

```
data/rules/
├── _core_rules.md              # Shared across all agents
├── ingest_rules.md             # PDF extraction patterns
├── method_extraction_rules.md  # Method identification patterns
├── code_generation_rules.md    # PyTorch code patterns
├── validation_rules.md       # Error detection & fixing
├── architecture_patterns.md    # Neural architecture templates
├── dataset_rules.md            # Dataset loading patterns
├── equation_rules.md           # Math-to-code conversion
└── conformal_prediction_rules.md  # Statistical validation
```

### 2.2 Core Rules Structure

Each rule file follows this structure:

```markdown
# Agent Rules: {Agent Name}

## Version
- Rules Version: 1.0.0
- Last Updated: 2026-03-17
- Compatible Models: gpt-oss:120b-cloud, deepseek-v3.1:671b-cloud

## Table of Contents
1. [Core Principles](#core-principles)
2. [Input Patterns](#input-patterns)
3. [Output Specifications](#output-specifications)
4. [Error Handling](#error-handling)
5. [Examples](#examples)
6. [Edge Cases](#edge-cases)
7. [Validation Rules](#validation-rules)

---

## Core Principles

### Principle 1: {Name}
{Detailed explanation}

### Principle 2: {Name}
{Detailed explanation}

## Input Patterns

### Pattern 1: {Pattern Name}
**Regex:** `{regex}`
**Description:** {When to match}
**Action:** {What to do}
**Example:**
```
Input: {example input}
Output: {expected output}
```

## Output Specifications

### Field: {field_name}
**Type:** {data type}
**Required:** {yes/no}
**Validation:** {validation rules}
**Examples:**
- Good: {example}
- Bad: {example}

## Examples

### Example 1: {Scenario}
**Input:**
```
{input text}
```

**Processing:**
{step-by-step reasoning}

**Output:**
```json
{expected output}
```

## Edge Cases

### Edge Case 1: {Scenario}
**Problem:** {description}
**Solution:** {how to handle}
**Example:** {code/example}
```

### 2.3 Example: Method Extraction Rules

```markdown
# Method Extraction Rules

## Core Principles

### Principle 1: Explicit Over Implicit
Only extract information explicitly stated in the paper.
Never infer hyperparameters not mentioned.
Never assume default values.

### Principle 2: Context Preservation
Always include surrounding context (±2 sentences) for validation.
Mark inferred values with confidence < 0.5.

### Principle 3: Multi-Source Verification
Cross-reference across:
- Abstract (high-level)
- Method section (detailed)
- Experiments section (actual values used)
- Tables (numerical values)

## Algorithm Name Patterns

### Pattern 1: Direct Declaration
**Regex:** `(?i)(?:we propose|we introduce|we present|called|named)\s+["']?([^"'.]+)["']?`
**Description:** Paper explicitly names their method
**Confidence:** 1.0
**Example:**
```
"We propose a novel architecture called Transformer-XL"
→ Algorithm: "Transformer-XL"
```

### Pattern 2: Comparative Reference
**Regex:** `(?i)(?:unlike|compared to|previous|existing)\s+(\w+),\s+(?:our|we|this)\s+(?:work|paper|method)`
**Description:** Method defined by comparison
**Confidence:** 0.7
**Requires:** Manual verification

## Training Configuration Patterns

### Optimizer Detection

#### Pattern 1: Explicit Statement
**Regex:** `(?i)(?:optimizer|optimized with|using)\s+(?:the\s+)?(Adam|SGD|RMSprop|Adagrad|Adadelta|AdamW)`
**Confidence:** 1.0
**Output Format:** {"optimizer": "{match}", "source": "explicit"}

#### Pattern 2: Learning Rate Context
**Regex:** `(?i)(?:learning rate|lr)\s*[:=]\s*(\d+\.?\d*)\s*(?:×\s*10\^(\d+))?`
**Confidence:** 1.0
**Normalization:** Convert scientific notation
**Example:**
```
"learning rate = 1e-4" → 0.0001
"lr: 5 × 10^-4" → 0.0005
```

### Batch Size Detection

#### Pattern 1: Standard Format
**Regex:** `(?i)(?:batch size|batch_size)\s*[:=]\s*(\d+)`
**Confidence:** 1.0

#### Pattern 2: Hardware Context
**Regex:** `(?i)(?:trained on|using)\s+(\d+)\s+(?:GPUs?|TPUs?)\s+.*?\s+(?:batch size of|with)\s+(\d+)`
**Confidence:** 0.9
**Calculation:** Total batch = GPUs × batch_per_gpu

## Architecture Patterns

### CNN Patterns

#### Pattern 1: Layer Sequence
**Regex:** `(?i)(\d+)\s*(?:×|x)\s*(conv\w*|convolution)`
**Description:** Repeated convolution blocks
**Example:**
```
"3 × Conv2D with 3×3 kernels"
→ {"layer_types": ["Conv2D", "Conv2D", "Conv2D"]}
```

#### Pattern 2: Residual Block
**Regex:** `(?i)(?:residual|resnet|skip connection)`
**Description:** Residual connection present
**Output:** {"key_components": ["ResidualBlock"]}

### Transformer Patterns

#### Pattern 1: Attention Mechanism
**Regex:** `(?i)(?:multi-head|multihead)\s+(?:self-)?attention\s+with\s+(\d+)\s+heads`
**Output:**
```json
{
  "layer_types": ["MultiHeadAttention"],
  "attention_type": "multi-head",
  "num_heads": {match}
}
```

#### Pattern 2: Position Encoding
**Regex:** `(?i)(?:positional|position)\s+(?:encoding|embedding)`
**Variants:**
- "sinusoidal positional encoding" → {"pe_type": "sinusoidal"}
- "learned position embeddings" → {"pe_type": "learned"}
- "rotary position embedding (RoPE)" → {"pe_type": "rope"}

## Dataset Patterns

### Standard Datasets

| Mention | Canonical Name | Loader |
|---------|---------------|--------|
| CIFAR-10, CIFAR10 | cifar-10 | torchvision.datasets.CIFAR10 |
| ImageNet, ILSVRC | imagenet | torchvision.datasets.ImageNet |
| MNIST | mnist | torchvision.datasets.MNIST |
| FER2013, FER-2013 | fer2013 | Custom loader |

### Pattern Matching

#### Fuzzy Matching Rules
1. Case-insensitive
2. Remove hyphens/underscores for comparison
3. Common abbreviations ("IMAGENET1K" → "imagenet")
4. Year variants ("ImageNet 2012" → "imagenet")

## Confidence Scoring Rules

### Score Calculation

| Evidence Type | Score | Source |
|--------------|-------|--------|
| Exact quote from text | 1.0 | "explicit" |
| Substring match | 0.85 | "explicit" |
| Pattern match | 0.7 | "pattern" |
| Contextual inference | 0.5 | "inferred" |
| No evidence | 0.3 | "low_confidence" |
| Contradicts text | 0.0 | "invalid" |

### Low Confidence Triggers

Flag for review when:
- Score < 0.5 for critical fields (algorithm_name, learning_rate)
- Multiple conflicting values found
- Value not in expected range (e.g., lr > 0.1)
- No supporting evidence in 3+ sentence window

## Error Handling

### Error Type 1: Ambiguous Reference
**Problem:** "We use the same setup as [1]"
**Solution:**
1. Check bibliography for [1]
2. If available, extract from cited paper
3. If not, mark as "reference_unavailable" with confidence 0.4

### Error Type 2: Range Values
**Problem:** "learning rate between 1e-4 and 1e-3"
**Solution:**
1. Extract range: [0.0001, 0.001]
2. Check experiments section for actual value used
3. If not found, use midpoint with confidence 0.5

## Validation Rules

### Post-Extraction Checks

1. **Consistency Check**
   - batch_size should be power of 2 (usually)
   - learning_rate should be < 0.1 (usually)
   - epochs should be > 0 and < 10000

2. **Completeness Check**
   - Critical fields: algorithm_name, optimizer, loss
   - Warning if missing: learning_rate, batch_size, epochs

3. **Cross-Reference Check**
   - Abstract claims vs Method details
   - Method description vs Experiments implementation
```

### 2.4 Rule Loader Implementation

```python
# src/rules/rule_loader.py

import re
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml

class RuleLoader:
    """
    Loads and parses rule files for agents.
    Supports markdown with embedded YAML/JSON blocks.
    """

    RULES_DIR = Path(__file__).parent

    @classmethod
    def load_rules(cls, agent_name: str) -> Dict[str, Any]:
        """
        Load rules for a specific agent.

        Returns structured rules with:
        - patterns: List of regex patterns
        - examples: List of input/output examples
        - validators: List of validation functions (as code)
        - edge_cases: List of edge case handlers
        """
        rule_file = cls.RULES_DIR / f"{agent_name}_rules.md"

        if not rule_file.exists():
            raise FileNotFoundError(f"Rules not found: {rule_file}")

        content = rule_file.read_text(encoding="utf-8")
        return cls._parse_rules(content)

    @classmethod
    def _parse_rules(cls, content: str) -> Dict[str, Any]:
        """Parse markdown rule file into structured format."""
        rules = {
            "version": None,
            "patterns": [],
            "examples": [],
            "validators": [],
            "edge_cases": [],
            "principles": []
        }

        # Extract version
        version_match = re.search(r'Rules Version:\s*(\S+)', content)
        if version_match:
            rules["version"] = version_match.group(1)

        # Extract patterns (### Pattern sections)
        pattern_sections = re.findall(
            r'### Pattern \d+:([^#]+?)(?=###|\Z)',
            content,
            re.DOTALL
        )

        for section in pattern_sections:
            pattern = cls._parse_pattern_section(section)
            if pattern:
                rules["patterns"].append(pattern)

        # Extract examples
        example_sections = re.findall(
            r'### Example \d+:([^#]+?)(?=###|\Z)',
            content,
            re.DOTALL
        )

        for section in example_sections:
            example = cls._parse_example_section(section)
            if example:
                rules["examples"].append(example)

        return rules

    @classmethod
    def _parse_pattern_section(cls, section: str) -> Optional[Dict]:
        """Parse a pattern section into structured format."""
        pattern = {}

        # Extract name
        name_match = re.match(r'\s*([^\n]+)', section)
        if name_match:
            pattern["name"] = name_match.group(1).strip()

        # Extract regex
        regex_match = re.search(r'\*\*Regex:\*\*\s*`([^`]+)`', section)
        if regex_match:
            pattern["regex"] = regex_match.group(1)

        # Extract confidence
        conf_match = re.search(r'\*\*Confidence:\*\*\s*(\d+\.?\d*)', section)
        if conf_match:
            pattern["confidence"] = float(conf_match.group(1))

        return pattern if pattern else None


# Usage in agents
class MethodExtractorAgent(BaseAgent):
    def __init__(self):
        super().__init__("method_extractor", "Method Extractor")
        self.rules = RuleLoader.load_rules("method_extraction")

    def process(self, message: AgentMessage) -> AgentResponse:
        # Load rules as context
        rules_context = self._format_rules_for_llm()

        prompt = f"""
        {rules_context}

        Now extract method information from this text:
        {message.payload['text']}

        Follow the patterns and validation rules above.
        """

        result = self.llm.chat(prompt)
        return AgentResponse(success=True, data=result)

    def _format_rules_for_llm(self) -> str:
        """Format rules for LLM context window."""
        lines = [
            "# EXTRACTION RULES",
            "",
            "## Patterns to Match:",
        ]

        for pattern in self.rules["patterns"]:
            lines.append(f"\n### {pattern['name']}")
            lines.append(f"Regex: `{pattern['regex']}`")
            lines.append(f"Confidence: {pattern['confidence']}")

        return "\n".join(lines)
```

---

## Phase 3: Proper Conformal Prediction (Week 3-4)

### 3.1 Statistical Implementation

```python
# src/conformal/predictor.py

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import json

@dataclass
class ConformalPrediction:
    """Result of conformal prediction."""
    prediction_set: List[Dict]  # Set of possible values
    point_estimate: Dict        # Best single prediction
    confidence: float           # 1 - alpha
    set_size: int               # |prediction_set|
    is_uncertain: bool          # set_size > 1
    calibration_size: int       # Size of calibration set

class ConformalPredictor:
    """
    Proper conformal prediction with coverage guarantees.

    Guarantees: P(true_value ∈ prediction_set) ≥ 1 - α
    """

    def __init__(self, alpha: float = 0.1):
        """
        Args:
            alpha: Miscoverage rate (e.g., 0.1 for 90% coverage)
        """
        self.alpha = alpha
        self.calibration_scores: Dict[str, List[float]] = {}
        self.quantiles: Dict[str, float] = {}
        self.is_calibrated = False

    def calibrate(
        self,
        validation_data: List[Tuple[str, Dict]],
        predictor_func
    ):
        """
        Calibrate using validation set.

        Args:
            validation_data: List of (text, true_structure) pairs
            predictor_func: Function that makes predictions
        """
        print(f"Calibrating conformal predictor with α={self.alpha}...")

        # Collect nonconformity scores for each field
        field_scores: Dict[str, List[float]] = {}

        for text, true_struct in validation_data:
            # Get prediction
            predicted = predictor_func(text)

            # Calculate nonconformity scores
            scores = self._calculate_nonconformity(predicted, true_struct)

            # Accumulate by field
            for field, score in scores.items():
                if field not in field_scores:
                    field_scores[field] = []
                field_scores[field].append(score)

        # Compute quantiles for each field
        n = len(validation_data)
        for field, scores in field_scores.items():
            # Adjusted quantile level for finite sample
            q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
            q_level = min(q_level, 1.0)  # Cap at 1.0

            self.quantiles[field] = np.quantile(scores, q_level)
            self.calibration_scores[field] = scores

        self.is_calibrated = True
        print(f"Calibration complete. Fields: {list(self.quantiles.keys())}")

    def _calculate_nonconformity(
        self,
        predicted: Dict,
        true: Dict
    ) -> Dict[str, float]:
        """
        Calculate nonconformity score for each field.
        Lower score = more conformal (better prediction).
        """
        scores = {}

        # Algorithm name - string similarity
        if "algorithm_name" in predicted and "algorithm_name" in true:
            scores["algorithm_name"] = self._string_nonconformity(
                predicted["algorithm_name"],
                true["algorithm_name"]
            )

        # Numeric fields - absolute difference
        numeric_fields = ["learning_rate", "batch_size", "epochs"]
        for field in numeric_fields:
            if field in predicted and field in true:
                pred_val = predicted[field]
                true_val = true[field]

                if pred_val is not None and true_val is not None:
                    scores[field] = self._numeric_nonconformity(pred_val, true_val)

        # List fields - Jaccard distance
        if "datasets" in predicted and "datasets" in true:
            scores["datasets"] = self._jaccard_nonconformity(
                predicted["datasets"],
                true["datasets"]
            )

        return scores

    def _string_nonconformity(self, pred: str, true: str) -> float:
        """Nonconformity for strings (0 = exact match, 1 = completely different)."""
        if pred == true:
            return 0.0

        # Use normalized Levenshtein distance
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, pred.lower(), true.lower()).ratio()
        return 1.0 - similarity

    def _numeric_nonconformity(self, pred: float, true: float) -> float:
        """Nonconformity for numeric values (normalized)."""
        if true == 0:
            return 0.0 if pred == 0 else 1.0

        relative_error = abs(pred - true) / abs(true)
        return min(relative_error, 1.0)  # Cap at 1.0

    def _jaccard_nonconformity(self, pred: List, true: List) -> float:
        """Jaccard distance for sets (1 - |intersection|/|union|)."""
        pred_set = set(pred)
        true_set = set(true)

        intersection = len(pred_set & true_set)
        union = len(pred_set | true_set)

        if union == 0:
            return 0.0

        return 1.0 - (intersection / union)

    def predict(
        self,
        text: str,
        predictor_func,
        candidate_generator_func
    ) -> Dict[str, ConformalPrediction]:
        """
        Make prediction with conformal guarantee.

        Args:
            text: Input text
            predictor_func: Function to get point prediction
            candidate_generator_func: Function to generate candidate set

        Returns:
            Dict mapping field names to ConformalPrediction objects
        """
        if not self.is_calibrated:
            raise RuntimeError("Must calibrate before prediction")

        # Get point prediction
        point_pred = predictor_func(text)

        # Generate candidate set
        candidates = candidate_generator_func(text, point_pred)

        # Build prediction sets for each field
        results = {}

        for field in self.quantiles.keys():
            q_hat = self.quantiles[field]

            # Include all candidates with score ≤ q_hat
            prediction_set = []

            for candidate in candidates:
                # Calculate nonconformity score
                score = self._score_candidate(field, candidate, point_pred)

                if score <= q_hat:
                    prediction_set.append(candidate)

            results[field] = ConformalPrediction(
                prediction_set=prediction_set,
                point_estimate=point_pred.get(field),
                confidence=1 - self.alpha,
                set_size=len(prediction_set),
                is_uncertain=len(prediction_set) > 1,
                calibration_size=len(self.calibration_scores.get(field, []))
            )

        return results

    def _score_candidate(self, field: str, candidate: Dict, point_pred: Dict) -> float:
        """Score a candidate against the point prediction."""
        # Create temporary prediction with this candidate value
        temp_pred = point_pred.copy()
        temp_pred[field] = candidate.get(field)

        # Calculate nonconformity
        scores = self._calculate_nonconformity(temp_pred, point_pred)
        return scores.get(field, 1.0)


# Integration with MethodExtractorAgent
class ConformalMethodExtractor:
    """Method extractor with conformal prediction."""

    def __init__(self):
        self.predictor = ConformalPredictor(alpha=0.1)  # 90% coverage
        self.base_extractor = BaseLLMExtractor()
        self.is_calibrated = False

    def calibrate(self, validation_papers: List[str]):
        """Calibrate on validation papers with known ground truth."""
        validation_data = []

        for paper in validation_papers:
            # Load ground truth (manually annotated)
            text = load_paper_text(paper)
            true_struct = load_ground_truth(paper)

            validation_data.append((text, true_struct))

        self.predictor.calibrate(
            validation_data,
            predictor_func=self.base_extractor.extract
        )

        self.is_calibrated = True

    def extract(self, text: str) -> Dict:
        """Extract with conformal prediction sets."""
        if not self.is_calibrated:
            print("Warning: Using uncalibrated predictor")
            return self.base_extractor.extract(text)

        # Get conformal predictions
        results = self.predictor.predict(
            text,
            predictor_func=self.base_extractor.extract,
            candidate_generator_func=self._generate_candidates
        )

        # Format output
        output = {
            "predictions": {},
            "uncertain_fields": [],
            "coverage_guarantee": f"{(1-self.predictor.alpha)*100}%"
        }

        for field, pred in results.items():
            output["predictions"][field] = {
                "value": pred.point_estimate,
                "prediction_set": pred.prediction_set if pred.is_uncertain else [pred.point_estimate],
                "confidence": pred.confidence,
                "is_uncertain": pred.is_uncertain
            }

            if pred.is_uncertain:
                output["uncertain_fields"].append(field)

        return output

    def _generate_candidates(self, text: str, point_pred: Dict) -> List[Dict]:
        """Generate candidate predictions by perturbing point prediction."""
        candidates = [point_pred]

        # Generate variants for numeric fields
        if "learning_rate" in point_pred:
            base_lr = point_pred["learning_rate"]
            if base_lr:
                # Try nearby values
                for factor in [0.1, 0.5, 2, 10]:
                    variant = point_pred.copy()
                    variant["learning_rate"] = base_lr * factor
                    candidates.append(variant)

        return candidates
```

### 3.2 Calibration Data Format

```json
{
  "calibration_papers": [
    {
      "paper_id": "transformer_vaswani_2017",
      "text_file": "transformer.txt",
      "ground_truth": {
        "algorithm_name": "Transformer",
        "architecture": {
          "layer_types": ["MultiHeadAttention", "FeedForward", "LayerNorm"],
          "attention_type": "multi-head",
          "num_heads": 8
        },
        "training": {
          "optimizer": "Adam",
          "learning_rate": 0.0001,
          "batch_size": 4096,
          "epochs": 100000
        },
        "datasets": ["WMT 2014 English-German", "WMT 2014 English-French"]
      }
    }
  ]
}
```

---

## Phase 4: Gemini Embeddings Integration (Week 4)

### 4.1 Implementation

```python
# src/embeddings/gemini_embedder.py

import os
from typing import List, Union
import numpy as np

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from sentence_transformers import SentenceTransformer


class GeminiEmbedder:
    """
    Gemini embedding model wrapper.
    Supports text, images, and documents.
    """

    MODELS = {
        "text": "gemini-embedding-001",
        "multimodal": "gemini-embedding-2-preview"
    }

    DIMENSIONS = {
        "small": 768,
        "medium": 1536,
        "large": 3072
    }

    def __init__(
        self,
        model_type: str = "text",
        output_dimensionality: int = 768,
        task_type: str = "RETRIEVAL_DOCUMENT"
    ):
        """
        Args:
            model_type: "text" or "multimodal"
            output_dimensionality: 768, 1536, or 3072
            task_type: See Gemini API docs
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("Google GenAI package not installed. Run: pip install google-genai")

        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        self.client = genai.Client(api_key=self.api_key)
        self.model = self.MODELS.get(model_type, "gemini-embedding-001")
        self.output_dimensionality = output_dimensionality
        self.task_type = task_type

    def embed(
        self,
        contents: Union[str, List[str]],
        is_document: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings.

        Args:
            contents: Text or list of texts
            is_document: True for documents, False for queries

        Returns:
            List of embedding vectors
        """
        if isinstance(contents, str):
            contents = [contents]

        # Determine task type
        task = self.task_type
        if not is_document and self.task_type == "RETRIEVAL_DOCUMENT":
            task = "RETRIEVAL_QUERY"

        # Batch processing (max 100 per request)
        all_embeddings = []
        batch_size = 100

        for i in range(0, len(contents), batch_size):
            batch = contents[i:i + batch_size]

            result = self.client.models.embed_content(
                model=self.model,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task,
                    output_dimensionality=self.output_dimensionality
                )
            )

            for embedding in result.embeddings:
                all_embeddings.append(embedding.values)

        return all_embeddings

    def embed_pdf(self, pdf_path: str) -> List[float]:
        """Embed a PDF document (multimodal model only)."""
        if self.model != "gemini-embedding-2-preview":
            raise ValueError("PDF embedding requires multimodal model")

        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()

        result = self.client.models.embed_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(
                    data=pdf_bytes,
                    mime_type='application/pdf'
                )
            ]
        )

        return result.embeddings[0].values


class HybridEmbedder:
    """
    Uses Gemini when available, falls back to local MiniLM.
    """

    def __init__(self, output_dimensionality: int = 768):
        self.gemini = None
        self.local = None
        self.output_dimensionality = output_dimensionality

        # Try Gemini first
        try:
            self.gemini = GeminiEmbedder(
                model_type="text",
                output_dimensionality=output_dimensionality
            )
            print(f"Using Gemini embeddings ({output_dimensionality}d)")
        except Exception as e:
            print(f"Gemini not available: {e}")
            print("Falling back to local MiniLM embeddings (384d)")
            self.local = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self.gemini:
            return self.gemini.embed(texts)
        else:
            embeddings = self.local.encode(texts)
            return embeddings.tolist()

    @property
    def dimensionality(self) -> int:
        if self.gemini:
            return self.output_dimensionality
        return 384  # MiniLM dimension


# Update index_documents.py to use hybrid embedder
# from embeddings.gemini_embedder import HybridEmbedder
# embedder = HybridEmbedder(output_dimensionality=768)
```

---

## Phase 5: NewResearcher Integration (Week 5)

### 5.1 Components to Port

```python
# src/integrations/newresearcher_components.py

"""
Ported components from NewResearcher (CrewAI-based).
Integrated into custom orchestrator.
"""

from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class TokenAwareChunker:
    """
    NewResearcher's token-based chunking.
    Ported and adapted for Research2Text.
    """

    MAX_CHUNK_TOKENS: int = 800
    OVERLAP_TOKENS: int = 100
    HARD_DOC_LIMIT: int = 3000  # chars

    def chunk(self, text: str) -> List[str]:
        """Token-aware sentence boundary chunking."""
        # Implementation from NewResearcher's text_chunker.py
        # ... (see earlier analysis)
        pass


class SourceValidator:
    """
    NewResearcher's source validation.
    Scores sources before extraction.
    """

    def validate_sources(
        self,
        sources: List[Dict],
        min_score: float = 7.0,
        max_sources: int = 5
    ) -> List[Dict]:
        """
        Score and filter sources.

        Scoring criteria:
        - Credibility (0-10): Is this a reputable source?
        - Recency (0-10): How recent is the publication?
        - Technical depth (0-10): Does it have implementation details?

        Returns top N sources above threshold.
        """
        scored = []

        for source in sources:
            score = self._score_source(source)
            if score >= min_score:
                scored.append({**source, "validation_score": score})

        # Sort by score descending
        scored.sort(key=lambda x: x["validation_score"], reverse=True)

        return scored[:max_sources]

    def _score_source(self, source: Dict) -> float:
        """Calculate composite score."""
        # Use LLM to score
        # ... implementation
        pass


class SearchToolIntegration:
    """
    Optional search tools from NewResearcher.
    """

    def __init__(self):
        self.exa_api_key = os.getenv("EXA_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")

    def search_arxiv(self, query: str, num_results: int = 5) -> List[Dict]:
        """Search arXiv papers."""
        # Implementation using Exa or arXiv API
        pass

    def search_ieee(self, query: str, num_results: int = 5) -> List[Dict]:
        """Search IEEE papers."""
        pass
```

---

## Summary: Implementation Timeline

| Week | Focus | Key Deliverables |
|------|-------|------------------|
| 1 | Windows Sandbox | `WindowsSandbox` class, restricted Python runner |
| 2 | Rule System Setup | Rule file structure, `RuleLoader`, 2 example rule files |
| 3 | Rule System Completion | All 8 rule files (10k+ lines each), agent integration |
| 4 | Conformal Prediction | `ConformalPredictor`, calibration data, integration |
| 5 | Gemini Embeddings | `GeminiEmbedder`, `HybridEmbedder`, ChromaDB integration |
| 6 | NewResearcher Integration | Token-aware chunking, source validation |
| 7 | Testing | Unit tests, integration tests, calibration validation |
| 8 | Documentation | API docs, deployment guide, examples |

---

## Key Design Decisions

1. **Windows Sandbox**: Use subprocess + restricted Python instead of Docker
2. **Rule System**: Markdown-based with embedded YAML for structured data
3. **Conformal Prediction**: Field-specific quantiles with coverage guarantees
4. **Embeddings**: Hybrid approach (Gemini preferred, MiniLM fallback)
5. **Token Management**: Skip (local Ollama), keep chunk size limits

---

## Success Metrics

| Metric | Before | Target |
|--------|--------|--------|
| Code validation success | ~60% | >85% |
| Knowledge graph nodes | 10-15 | 40-60 |
| Extraction confidence | Heuristic | Statistical guarantee |
| Embedding quality | MTEB ~62 | MTEB ~68+ |
| Sandbox security | None | Process isolation + resource limits |

---

*Plan Version: 1.0*
*Date: 2026-03-17*

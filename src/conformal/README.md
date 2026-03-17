# Conformal Prediction Module

Statistical uncertainty quantification for method extraction with coverage guarantees.

## Overview

This module implements **Conformal Prediction** as described in:
> "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification"
> by Angelopoulos and Bates (2021)

**Key Property:** `P(true_value ∈ prediction_set) ≥ 1 - α`

For example, with α=0.1 (90% coverage), the prediction set is guaranteed to contain
the true value at least 90% of the time.

## Components

### Core Classes

- **`ConformalPredictor`** - Main predictor with calibration and prediction methods
- **`ConformalMethodExtractor`** - Wrapper for method extraction with conformal prediction
- **`ConformalPrediction`** - Result dataclass for individual field predictions
- **`ConformalMethodResult`** - Complete result for method extraction

### Nonconformity Scores

Per-field scoring functions:
- **String fields** (algorithm_name): Normalized Levenshtein distance
- **Numeric fields** (learning_rate, batch_size, epochs): Relative error
- **List fields** (datasets): Jaccard distance
- **Architecture fields**: Combined string and list metrics

## Usage

### Basic Usage

```python
from src.conformal import ConformalPredictor, create_conformal_extractor

# Create predictor with 90% coverage guarantee
predictor = ConformalPredictor(alpha=0.1)

# Calibrate on validation data
predictor.calibrate(validation_data, base_predictor)

# Make predictions with uncertainty quantification
result = predictor.predict(text, base_predictor)

# Access prediction sets
for field, pred in result.predictions.items():
    print(f"{field}: {pred.prediction_set} (confidence: {pred.confidence})")
    if pred.is_uncertain:
        print(f"  Warning: {field} has multiple possible values")
```

### With Method Extractor

```python
from src.conformal import ConformalMethodExtractor

# Wrap existing method extractor
conformal_extractor = ConformalMethodExtractor(
    base_extractor=method_extractor,
    alpha=0.1
)

# Calibrate on validation papers
conformal_extractor.calibrate(validation_papers)

# Extract with conformal prediction
result = conformal_extractor.extract(paper_text)

# Check uncertain fields
print(f"Uncertain fields: {result.uncertain_fields}")
print(f"Overall confidence: {result.overall_confidence:.2%}")
```

## Calibration Data

### Format

Calibration data is stored in `data/calibration/validation_papers.json`:

```json
{
  "coverage_target": 0.9,
  "papers": [
    {
      "paper_id": "cal_001",
      "text_file": "cal_001.txt",
      "ground_truth": {
        "algorithm_name": "CNN-BiLSTM",
        "datasets": ["CIFAR-10", "MNIST"],
        "training": {
          "learning_rate": 0.001,
          "batch_size": 32,
          "epochs": 100
        },
        "architecture": {
          "layer_types": ["Conv2D", "LSTM", "Dense"],
          "num_classes": 10
        }
      }
    }
  ]
}
```

### Generating Calibration Data

```bash
# Generate synthetic calibration data
python -m src.conformal.generate_calibration --synthetic --num-synthetic 20

# Extract from existing outputs
python -m src.conformal.generate_calibration --outputs-dir data/outputs
```

## Evaluation

Run coverage validation:

```bash
python -m src.conformal.evaluate \
    --calibration-file data/calibration/validation_papers.json \
    --alpha 0.1 \
    --output-dir outputs/conformal_eval \
    --plot
```

### Metrics

- **Coverage Rate**: Percentage of true values in prediction sets
- **Set Size**: Average number of candidates in prediction sets
- **Coverage Gap**: Difference between target and actual coverage

### Success Criteria

- Coverage rate ≥ 90% (for α=0.1)
- Average set size < 3 for most fields
- Better uncertainty quantification than heuristic confidence

## Integration with MethodExtractorAgent

The `MethodExtractorAgent` uses conformal prediction for confidence scoring:

```python
# Per-field confidence scores
confidence = {
    "algorithm_name": ConfidenceScore(
        score=0.95,
        source="explicit",
        evidence="Found exact match in text"
    ),
    "training.learning_rate": ConfidenceScore(
        score=0.85,
        source="explicit",
        evidence="Value '0.001' found in text"
    )
}
```

Confidence levels:
- **1.0**: Directly quoted from text
- **0.85**: Substring found in text
- **0.6**: Field type mentioned but value unclear
- **0.3**: Appears inferred/hallucinated
- **0.0**: Field is null/empty

## Files

- `predictor.py` - Core conformal prediction implementation
- `evaluate.py` - Evaluation and validation scripts
- `generate_calibration.py` - Calibration data generation
- `__init__.py` - Module exports
- `README.md` - This file

## References

1. Angelopoulos, A. N., & Bates, S. (2021). A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification. arXiv:2107.07511.
2. Shafer, G., & Vovk, V. (2008). A Tutorial on Conformal Prediction. Journal of Machine Learning Research, 9, 371-421.

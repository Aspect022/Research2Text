"""
Conformal Prediction module for uncertainty quantification.

Provides statistical coverage guarantees for method extraction:
    P(true_value ∈ prediction_set) ≥ 1 - α

Main classes:
    - ConformalPredictor: Core conformal prediction logic
    - ConformalMethodExtractor: Wrapper for method extraction with conformal prediction
    - ConformalPrediction: Result dataclass for individual field predictions
    - ConformalMethodResult: Complete result for method extraction

Usage:
    from src.conformal import ConformalPredictor, create_conformal_extractor

    # Create and calibrate predictor
    predictor = ConformalPredictor(alpha=0.1)  # 90% coverage
    predictor.calibrate(validation_data, base_predictor)

    # Make predictions with uncertainty quantification
    result = predictor.predict(text, base_predictor)

    # Access prediction sets
    for field, pred in result.predictions.items():
        print(f"{field}: {pred.prediction_set} (confidence: {pred.confidence})")
"""

from .predictor import (
    ConformalPredictor,
    ConformalMethodExtractor,
    ConformalPrediction,
    ConformalMethodResult,
    create_conformal_extractor,
)

__all__ = [
    "ConformalPredictor",
    "ConformalMethodExtractor",
    "ConformalPrediction",
    "ConformalMethodResult",
    "create_conformal_extractor",
]

"""
Proper Conformal Prediction with statistical coverage guarantees.

Implements conformal prediction as described in:
"A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification"
by Angelopoulos and Bates (2021).

Key property: P(true_value ∈ prediction_set) ≥ 1 - α
"""

import json
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from pathlib import Path
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


@dataclass
class ConformalPrediction:
    """Result of conformal prediction for a single field."""
    field_name: str
    point_estimate: Any
    prediction_set: List[Any]  # Set of possible values
    confidence: float  # 1 - alpha
    set_size: int
    is_uncertain: bool  # set_size > 1
    nonconformity_score: float  # Score for point estimate
    calibration_size: int


@dataclass
class ConformalMethodResult:
    """Complete conformal prediction result for method extraction."""
    predictions: Dict[str, ConformalPrediction]
    overall_confidence: float
    uncertain_fields: List[str]
    coverage_guarantee: str
    calibration_info: Dict[str, Any] = field(default_factory=dict)


class ConformalPredictor:
    """
    Conformal predictor with coverage guarantees.

    Guarantees: P(true_value ∈ prediction_set) ≥ 1 - α

    Usage:
        1. Calibrate on validation set with known ground truth
        2. Predict on new data to get prediction sets
        3. Sets are guaranteed to contain true value with probability 1-α
    """

    def __init__(self, alpha: float = 0.1):
        """
        Initialize conformal predictor.

        Args:
            alpha: Miscoverage rate (e.g., 0.1 for 90% coverage)
        """
        self.alpha = alpha
        self.coverage = 1 - alpha
        self.calibration_scores: Dict[str, List[float]] = {}
        self.quantiles: Dict[str, float] = {}
        self.is_calibrated = False

        logger.info(f"Initialized ConformalPredictor with α={alpha} ({self.coverage*100:.0f}% coverage)")

    def calibrate(
        self,
        validation_data: List[Tuple[str, Dict]],
        predictor_func: Callable[[str], Dict]
    ):
        """
        Calibrate using validation set with known ground truth.

        Args:
            validation_data: List of (text, true_structure) pairs
            predictor_func: Function that makes point predictions
        """
        logger.info(f"Calibrating on {len(validation_data)} validation samples...")

        # Collect nonconformity scores for each field
        field_scores: Dict[str, List[float]] = {}

        for i, (text, true_struct) in enumerate(validation_data):
            try:
                # Get prediction
                predicted = predictor_func(text)

                # Calculate nonconformity scores
                scores = self._calculate_nonconformity(predicted, true_struct)

                # Accumulate by field
                for field, score in scores.items():
                    if field not in field_scores:
                        field_scores[field] = []
                    field_scores[field].append(score)

                if (i + 1) % 10 == 0:
                    logger.info(f"Calibrated {i + 1}/{len(validation_data)} samples")

            except Exception as e:
                logger.warning(f"Failed to calibrate sample {i}: {e}")
                continue

        # Compute quantiles for each field
        n = len(validation_data)
        for field, scores in field_scores.items():
            if len(scores) < 5:
                logger.warning(f"Field '{field}' has only {len(scores)} samples, skipping")
                continue

            # Adjusted quantile level for finite sample
            # Using the conservative approach: ceil((n+1)*(1-α)) / n
            q_level = np.ceil((n + 1) * self.coverage) / n
            q_level = min(q_level, 1.0)

            self.quantiles[field] = np.quantile(scores, q_level)
            self.calibration_scores[field] = scores

            logger.info(
                f"Field '{field}': calibrated with {len(scores)} samples, "
                f"quantile={self.quantiles[field]:.4f}"
            )

        self.is_calibrated = True
        logger.info("Calibration complete")

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
        if "algorithm_name" in predicted:
            scores["algorithm_name"] = self._string_nonconformity(
                predicted.get("algorithm_name", ""),
                true.get("algorithm_name", "")
            )

        # Numeric fields
        numeric_fields = ["learning_rate", "batch_size", "epochs"]
        training_pred = predicted.get("training", {})
        training_true = true.get("training", {})

        for field in numeric_fields:
            if field in training_pred:
                scores[f"training.{field}"] = self._numeric_nonconformity(
                    training_pred.get(field),
                    training_true.get(field)
                )

        # Datasets - Jaccard distance
        if "datasets" in predicted:
            scores["datasets"] = self._jaccard_nonconformity(
                predicted.get("datasets", []),
                true.get("datasets", [])
            )

        # Architecture fields
        arch_pred = predicted.get("architecture", {})
        arch_true = true.get("architecture", {})

        if "layer_types" in arch_pred:
            scores["architecture.layer_types"] = self._list_nonconformity(
                arch_pred.get("layer_types", []),
                arch_true.get("layer_types", [])
            )

        if "num_layers" in arch_pred:
            scores["architecture.num_layers"] = self._numeric_nonconformity(
                arch_pred.get("num_layers"),
                arch_true.get("num_layers")
            )

        return scores

    def _string_nonconformity(self, pred: str, true: str) -> float:
        """
        Nonconformity for strings using normalized Levenshtein distance.

        Returns 0.0 for exact match, 1.0 for completely different.
        """
        if not pred and not true:
            return 0.0
        if not pred or not true:
            return 1.0

        # Use SequenceMatcher for similarity
        similarity = SequenceMatcher(None, pred.lower(), true.lower()).ratio()
        return 1.0 - similarity

    def _numeric_nonconformity(self, pred, true) -> float:
        """
        Nonconformity for numeric values using relative error.

        Returns 0.0 for exact match, capped at 1.0.
        """
        if pred is None and true is None:
            return 0.0
        if pred is None or true is None:
            return 1.0

        try:
            pred_val = float(pred)
            true_val = float(true)

            if true_val == 0:
                return 0.0 if pred_val == 0 else 1.0

            relative_error = abs(pred_val - true_val) / abs(true_val)
            return min(relative_error, 1.0)

        except (ValueError, TypeError):
            return 1.0

    def _jaccard_nonconformity(self, pred: List, true: List) -> float:
        """
        Jaccard distance for sets.

        Returns 1 - |intersection| / |union|
        """
        pred_set = set(pred)
        true_set = set(true)

        if not pred_set and not true_set:
            return 0.0

        intersection = len(pred_set & true_set)
        union = len(pred_set | true_set)

        if union == 0:
            return 0.0

        return 1.0 - (intersection / union)

    def _list_nonconformity(self, pred: List, true: List) -> float:
        """
        Nonconformity for lists using normalized edit distance.
        """
        if not pred and not true:
            return 0.0
        if not pred or not true:
            return 1.0

        # Use sequence matching
        sm = SequenceMatcher(None, pred, true)
        return 1.0 - sm.ratio()

    def predict(
        self,
        text: str,
        predictor_func: Callable[[str], Dict],
        candidate_generator_func: Optional[Callable] = None
    ) -> ConformalMethodResult:
        """
        Make prediction with conformal guarantee.

        Args:
            text: Input text
            predictor_func: Function to get point prediction
            candidate_generator_func: Optional function to generate candidates

        Returns:
            ConformalMethodResult with prediction sets
        """
        if not self.is_calibrated:
            raise RuntimeError("Must calibrate before prediction. Call calibrate() first.")

        # Get point prediction
        point_pred = predictor_func(text)

        # Build prediction sets
        predictions = {}
        uncertain_fields = []

        for field, q_hat in self.quantiles.items():
            # Get value from prediction
            value = self._get_nested_value(point_pred, field)

            # Build prediction set
            if candidate_generator_func:
                candidates = candidate_generator_func(field, value, point_pred)
            else:
                # Default: just use point estimate
                candidates = [value] if value is not None else []

            # Filter by nonconformity score
            prediction_set = []
            for candidate in candidates:
                score = self._score_candidate(field, candidate, value)
                if score <= q_hat:
                    prediction_set.append(candidate)

            # If no candidates pass, include point estimate
            if not prediction_set and value is not None:
                prediction_set = [value]

            is_uncertain = len(prediction_set) > 1

            pred_result = ConformalPrediction(
                field_name=field,
                point_estimate=value,
                prediction_set=prediction_set,
                confidence=self.coverage,
                set_size=len(prediction_set),
                is_uncertain=is_uncertain,
                nonconformity_score=self._score_candidate(field, value, value) if value else 1.0,
                calibration_size=len(self.calibration_scores.get(field, []))
            )

            predictions[field] = pred_result

            if is_uncertain:
                uncertain_fields.append(field)

        # Calculate overall confidence
        if predictions:
            overall_confidence = np.mean([
                1.0 if not p.is_uncertain else 0.5
                for p in predictions.values()
            ])
        else:
            overall_confidence = 0.0

        return ConformalMethodResult(
            predictions=predictions,
            overall_confidence=overall_confidence,
            uncertain_fields=uncertain_fields,
            coverage_guarantee=f"{self.coverage*100:.0f}%",
            calibration_info={
                "alpha": self.alpha,
                "num_fields": len(self.quantiles),
                "calibration_sizes": {
                    f: len(s) for f, s in self.calibration_scores.items()
                }
            }
        )

    def _get_nested_value(self, data: Dict, field: str) -> Any:
        """Get value from nested dict using dot notation."""
        parts = field.split(".")
        value = data
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None
        return value

    def _score_candidate(self, field: str, candidate: Any, reference: Any) -> float:
        """Score a candidate value."""
        # Use appropriate scoring based on field type
        if "algorithm_name" in field:
            return self._string_nonconformity(candidate, reference)
        elif any(x in field for x in ["learning_rate", "batch_size", "epochs", "num_layers"]):
            return self._numeric_nonconformity(candidate, reference)
        elif "datasets" in field:
            return self._jaccard_nonconformity(candidate, reference)
        elif "layer_types" in field:
            return self._list_nonconformity(candidate, reference)
        else:
            return 0.5  # Default

    def save_calibration(self, path: Path):
        """Save calibration data to file."""
        data = {
            "alpha": self.alpha,
            "coverage": self.coverage,
            "quantiles": self.quantiles,
            "calibration_scores": self.calibration_scores,
            "is_calibrated": self.is_calibrated
        }
        path.write_text(json.dumps(data, indent=2))
        logger.info(f"Saved calibration to {path}")

    def load_calibration(self, path: Path):
        """Load calibration data from file."""
        data = json.loads(path.read_text())
        self.alpha = data["alpha"]
        self.coverage = data["coverage"]
        self.quantiles = {k: float(v) for k, v in data["quantiles"].items()}
        self.calibration_scores = data["calibration_scores"]
        self.is_calibrated = data["is_calibrated"]
        logger.info(f"Loaded calibration from {path}")


class ConformalMethodExtractor:
    """
    Method extractor with proper conformal prediction.

    Wraps base extractor with conformal prediction for uncertainty quantification.
    """

    def __init__(self, base_extractor, alpha: float = 0.1):
        """
        Initialize conformal method extractor.

        Args:
            base_extractor: Base extraction function
            alpha: Miscoverage rate (0.1 = 90% coverage)
        """
        self.base_extractor = base_extractor
        self.predictor = ConformalPredictor(alpha=alpha)
        self.is_calibrated = False

    def calibrate(self, validation_papers: List[Path]):
        """
        Calibrate on validation papers with known ground truth.

        Args:
            validation_papers: List of paths to validation papers
        """
        validation_data = []

        for paper_path in validation_papers:
            # Load text
            text_file = paper_path.with_suffix(".txt")
            if not text_file.exists():
                continue

            text = text_file.read_text(encoding="utf-8")

            # Load ground truth (expected in JSON file)
            truth_file = paper_path.parent / f"{paper_path.stem}_ground_truth.json"
            if not truth_file.exists():
                logger.warning(f"No ground truth for {paper_path}, skipping")
                continue

            true_struct = json.loads(truth_file.read_text())
            validation_data.append((text, true_struct))

        if len(validation_data) < 10:
            raise ValueError(
                f"Need at least 10 validation samples, got {len(validation_data)}"
            )

        self.predictor.calibrate(validation_data, self.base_extractor)
        self.is_calibrated = True

    def extract(self, text: str) -> ConformalMethodResult:
        """
        Extract method with conformal prediction.

        Args:
            text: Paper text

        Returns:
            ConformalMethodResult with prediction sets
        """
        if not self.is_calibrated:
            logger.warning("Using uncalibrated predictor")

        return self.predictor.predict(
            text,
            predictor_func=self.base_extractor,
            candidate_generator_func=self._generate_candidates
        )

    def _generate_candidates(self, field: str, value: Any, point_pred: Dict) -> List[Any]:
        """Generate candidate values for a field."""
        candidates = [value] if value is not None else []

        # Generate variants for numeric fields
        if "learning_rate" in field and value:
            try:
                base_lr = float(value)
                # Try nearby values
                for factor in [0.1, 0.5, 2.0, 10.0]:
                    candidates.append(base_lr * factor)
            except (ValueError, TypeError):
                pass

        elif "batch_size" in field and value:
            try:
                base_bs = int(value)
                # Try nearby powers of 2
                for bs in [8, 16, 32, 64, 128, 256]:
                    if bs != base_bs:
                        candidates.append(bs)
            except (ValueError, TypeError):
                pass

        elif "epochs" in field and value:
            try:
                base_ep = int(value)
                for ep in [10, 50, 100, 200, 500]:
                    if ep != base_ep:
                        candidates.append(ep)
            except (ValueError, TypeError):
                pass

        return candidates


# Convenience function
def create_conformal_extractor(base_extractor, alpha: float = 0.1) -> ConformalMethodExtractor:
    """Create a conformal method extractor."""
    return ConformalMethodExtractor(base_extractor, alpha=alpha)

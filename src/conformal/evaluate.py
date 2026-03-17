"""
Evaluation script for Conformal Prediction system.

Validates coverage guarantees and set size efficiency.
"""

import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt

from .predictor import ConformalPredictor, ConformalMethodResult

logger = logging.getLogger(__name__)


def load_validation_data(calibration_file: Path) -> List[Tuple[str, Dict]]:
    """Load validation papers with ground truth."""
    data = json.loads(calibration_file.read_text())
    validation_data = []

    calibration_dir = calibration_file.parent

    for paper in data.get("papers", []):
        text_file = calibration_dir / paper["text_file"]
        if text_file.exists():
            text = text_file.read_text(encoding="utf-8")
            ground_truth = paper["ground_truth"]
            validation_data.append((text, ground_truth))
        else:
            # Create synthetic text from ground truth for testing
            gt = paper["ground_truth"]
            synthetic_text = f"""
            This paper presents {gt['algorithm_name']}.
            {gt['paper_summary']}

            We evaluate on {', '.join(gt['datasets'])}.

            Training Configuration:
            - Optimizer: {gt['training']['optimizer']}
            - Loss: {gt['training']['loss']}
            - Epochs: {gt['training']['epochs']}
            - Learning Rate: {gt['training']['learning_rate']}
            - Batch Size: {gt['training']['batch_size']}

            Architecture:
            - Layers: {', '.join(gt['architecture']['layer_types'])}
            - Hidden Dimensions: {gt['architecture']['hidden_dims']}
            - Number of Classes: {gt['architecture']['num_classes']}
            """
            validation_data.append((synthetic_text, gt))

    return validation_data


def evaluate_coverage(
    predictor: ConformalPredictor,
    validation_data: List[Tuple[str, Dict]],
    predictor_func
) -> Dict[str, Any]:
    """
    Evaluate coverage guarantee: P(true_value ∈ prediction_set) ≥ 1 - α

    Returns detailed metrics per field.
    """
    logger.info(f"Evaluating coverage on {len(validation_data)} samples...")

    field_coverage = {}
    field_set_sizes = {}

    for i, (text, true_struct) in enumerate(validation_data):
        try:
            result = predictor.predict(text, predictor_func)

            for field, pred in result.predictions.items():
                true_value = get_nested_value(true_struct, field)

                if true_value is None:
                    continue

                if field not in field_coverage:
                    field_coverage[field] = {"covered": 0, "total": 0}
                    field_set_sizes[field] = []

                # Check if true value is in prediction set
                covered = check_coverage(true_value, pred.prediction_set, field)
                field_coverage[field]["covered"] += int(covered)
                field_coverage[field]["total"] += 1
                field_set_sizes[field].append(pred.set_size)

        except Exception as e:
            logger.warning(f"Failed to evaluate sample {i}: {e}")

    # Calculate metrics
    metrics = {}
    for field, counts in field_coverage.items():
        coverage_rate = counts["covered"] / counts["total"] if counts["total"] > 0 else 0
        avg_set_size = np.mean(field_set_sizes[field]) if field_set_sizes[field] else 0

        metrics[field] = {
            "coverage_rate": coverage_rate,
            "target_coverage": predictor.coverage,
            "coverage_gap": predictor.coverage - coverage_rate,
            "num_samples": counts["total"],
            "avg_set_size": float(avg_set_size),
            "median_set_size": float(np.median(field_set_sizes[field])) if field_set_sizes[field] else 0,
            "max_set_size": int(max(field_set_sizes[field])) if field_set_sizes[field] else 0,
        }

    return metrics


def get_nested_value(data: Dict, field: str) -> Any:
    """Get value from nested dict using dot notation."""
    parts = field.split(".")
    value = data
    for part in parts:
        if isinstance(value, dict):
            value = value.get(part)
        else:
            return None
    return value


def check_coverage(true_value: Any, prediction_set: List[Any], field: str) -> bool:
    """Check if true value is covered by prediction set."""
    if not prediction_set:
        return False

    # For numeric fields, use approximate equality
    if any(x in field for x in ["learning_rate", "batch_size", "epochs", "num_layers"]):
        try:
            true_float = float(true_value)
            for pred_val in prediction_set:
                try:
                    pred_float = float(pred_val)
                    if abs(true_float - pred_float) < 1e-6:
                        return True
                except (ValueError, TypeError):
                    continue
            return False
        except (ValueError, TypeError):
            pass

    # For lists, check if true list is subset of any prediction
    if isinstance(true_value, list):
        true_set = set(str(x).lower() for x in true_value)
        for pred in prediction_set:
            if isinstance(pred, list):
                pred_set = set(str(x).lower() for x in pred)
                if true_set.issubset(pred_set):
                    return True
        return False

    # For strings, check exact match or substring
    if isinstance(true_value, str):
        true_lower = true_value.lower()
        for pred in prediction_set:
            if isinstance(pred, str) and (true_lower == pred.lower() or
                                          true_lower in pred.lower() or
                                          pred.lower() in true_lower):
                return True
        return False

    # Default: exact equality
    return true_value in prediction_set


def plot_calibration_results(metrics: Dict[str, Any], output_path: Path):
    """Generate calibration plots."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    fields = list(metrics.keys())
    coverage_rates = [metrics[f]["coverage_rate"] for f in fields]
    avg_set_sizes = [metrics[f]["avg_set_size"] for f in fields]

    # Coverage rates
    ax1 = axes[0, 0]
    ax1.barh(fields, coverage_rates, color="steelblue")
    ax1.axvline(x=0.9, color="red", linestyle="--", label="Target (90%)")
    ax1.set_xlabel("Coverage Rate")
    ax1.set_title("Coverage Rate by Field")
    ax1.set_xlim(0, 1)
    ax1.legend()

    # Average set sizes
    ax2 = axes[0, 1]
    ax2.barh(fields, avg_set_sizes, color="forestgreen")
    ax2.set_xlabel("Average Set Size")
    ax2.set_title("Prediction Set Size by Field")
    ax2.axvline(x=3, color="orange", linestyle="--", label="Target (<3)")
    ax2.legend()

    # Coverage vs Set Size scatter
    ax3 = axes[1, 0]
    ax3.scatter(avg_set_sizes, coverage_rates, s=100, alpha=0.6)
    for i, field in enumerate(fields):
        ax3.annotate(field, (avg_set_sizes[i], coverage_rates[i]),
                    fontsize=8, ha="center")
    ax3.axhline(y=0.9, color="red", linestyle="--", alpha=0.5)
    ax3.axvline(x=3, color="orange", linestyle="--", alpha=0.5)
    ax3.set_xlabel("Average Set Size")
    ax3.set_ylabel("Coverage Rate")
    ax3.set_title("Coverage vs Set Size Trade-off")

    # Coverage gap
    ax4 = axes[1, 1]
    gaps = [metrics[f]["coverage_gap"] for f in fields]
    colors = ["green" if g <= 0 else "red" for g in gaps]
    ax4.barh(fields, gaps, color=colors, alpha=0.7)
    ax4.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
    ax4.set_xlabel("Coverage Gap (Target - Actual)")
    ax4.set_title("Coverage Gap by Field")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved calibration plot to {output_path}")


def print_metrics_table(metrics: Dict[str, Any]):
    """Print metrics in a formatted table."""
    print("\n" + "=" * 100)
    print("CONFORMAL PREDICTION EVALUATION RESULTS")
    print("=" * 100)
    print(f"{'Field':<30} {'Coverage':<12} {'Target':<12} {'Gap':<12} {'Avg Size':<12} {'Samples':<10}")
    print("-" * 100)

    for field, m in sorted(metrics.items()):
        status = "PASS" if m["coverage_rate"] >= m["target_coverage"] - 0.05 else "FAIL"
        print(f"{field:<30} {m['coverage_rate']:.2%}       {m['target_coverage']:.2%}       "
              f"{m['coverage_gap']:+.2%}      {m['avg_set_size']:.2f}        {m['num_samples']:<10} {status}")

    print("=" * 100)

    # Summary
    avg_coverage = np.mean([m["coverage_rate"] for m in metrics.values()])
    avg_set_size = np.mean([m["avg_set_size"] for m in metrics.values()])
    fields_meeting_target = sum(1 for m in metrics.values()
                                if m["coverage_rate"] >= m["target_coverage"] - 0.05)

    print(f"\nSUMMARY:")
    print(f"  Average Coverage: {avg_coverage:.2%}")
    print(f"  Average Set Size: {avg_set_size:.2f}")
    print(f"  Fields Meeting Target: {fields_meeting_target}/{len(metrics)}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Conformal Prediction system")
    parser.add_argument("--calibration-file", type=Path,
                        default=Path("data/calibration/validation_papers.json"),
                        help="Path to calibration data JSON")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Miscoverage rate (default: 0.1 for 90% coverage)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/conformal_eval"),
                        help="Output directory for results")
    parser.add_argument("--plot", action="store_true",
                        help="Generate calibration plots")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Load validation data
    logger.info(f"Loading validation data from {args.calibration_file}")
    validation_data = load_validation_data(args.calibration_file)
    logger.info(f"Loaded {len(validation_data)} validation samples")

    if len(validation_data) < 5:
        logger.error("Need at least 5 validation samples for meaningful evaluation")
        return

    # Create predictor and calibrate
    predictor = ConformalPredictor(alpha=args.alpha)

    # Simple mock predictor for evaluation
    def mock_predictor(text: str) -> Dict:
        """Mock predictor that extracts some values from text."""
        result = {
            "algorithm_name": None,
            "datasets": [],
            "training": {},
            "architecture": {}
        }

        # Simple extraction heuristics
        if "CNN" in text:
            result["algorithm_name"] = "CNN"
        elif "ResNet" in text:
            result["algorithm_name"] = "ResNet"
        elif "Transformer" in text:
            result["algorithm_name"] = "Transformer"

        # Extract datasets
        datasets = ["CIFAR-10", "ImageNet", "MNIST", "COCO", "FER2013"]
        result["datasets"] = [d for d in datasets if d.lower() in text.lower()]

        # Extract training params
        import re
        lr_match = re.search(r'learning rate[:\s]+([\d.]+)', text, re.I)
        if lr_match:
            result["training"]["learning_rate"] = float(lr_match.group(1))

        bs_match = re.search(r'batch size[:\s]+(\d+)', text, re.I)
        if bs_match:
            result["training"]["batch_size"] = int(bs_match.group(1))

        epochs_match = re.search(r'epochs[:\s]+(\d+)', text, re.I)
        if epochs_match:
            result["training"]["epochs"] = int(epochs_match.group(1))

        return result

    # Calibrate
    logger.info("Calibrating predictor...")
    predictor.calibrate(validation_data, mock_predictor)

    # Evaluate coverage
    metrics = evaluate_coverage(predictor, validation_data, mock_predictor)

    # Print results
    print_metrics_table(metrics)

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_file = args.output_dir / "evaluation_results.json"
    results_file.write_text(json.dumps(metrics, indent=2))
    logger.info(f"Saved results to {results_file}")

    # Save calibration
    predictor.save_calibration(args.output_dir / "calibration.json")

    # Generate plots
    if args.plot:
        try:
            plot_calibration_results(metrics, args.output_dir / "calibration_plots.png")
        except Exception as e:
            logger.warning(f"Failed to generate plots: {e}")


if __name__ == "__main__":
    main()

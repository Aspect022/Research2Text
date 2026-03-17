"""
Generate calibration data from existing processed papers.

This script extracts ground truth from papers that have been manually verified
or from synthetic data to create calibration datasets for conformal prediction.
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


def extract_from_existing_outputs(outputs_dir: Path) -> List[Dict[str, Any]]:
    """
    Extract calibration data from existing paper outputs.

    Uses method.json files from previously processed papers.
    """
    calibration_papers = []

    for method_file in outputs_dir.rglob("method.json"):
        try:
            data = json.loads(method_file.read_text())

            # Extract relevant fields
            paper_id = method_file.parent.name.replace(" ", "_").replace("-", "_")[:50]

            calibration_entry = {
                "paper_id": f"auto_{paper_id}",
                "title": data.get("algorithm_name", "Unknown"),
                "text_file": f"{paper_id}.txt",
                "ground_truth": {
                    "algorithm_name": data.get("algorithm_name"),
                    "paper_summary": data.get("paper_summary"),
                    "equations": data.get("equations", []),
                    "datasets": data.get("datasets", []),
                    "training": data.get("training", {}),
                    "architecture": data.get("architecture", {}),
                    "inputs": data.get("inputs", {}),
                    "outputs": data.get("outputs", {}),
                    "references": data.get("references", [])
                }
            }

            calibration_papers.append(calibration_entry)
            logger.info(f"Extracted calibration data from {method_file}")

        except Exception as e:
            logger.warning(f"Failed to process {method_file}: {e}")

    return calibration_papers


def create_synthetic_calibration_data(num_samples: int = 10) -> List[Dict[str, Any]]:
    """Create synthetic calibration data for testing."""
    templates = [
        {
            "algorithm_name": "Convolutional Neural Network",
            "datasets": ["CIFAR-10", "MNIST"],
            "training": {
                "optimizer": "Adam",
                "loss": "CrossEntropyLoss",
                "epochs": 50,
                "learning_rate": 0.001,
                "batch_size": 32
            },
            "architecture": {
                "layer_types": ["Conv2D", "ReLU", "MaxPool", "Flatten", "Dense"],
                "num_classes": 10,
                "hidden_dims": [128, 64]
            }
        },
        {
            "algorithm_name": "Recurrent Neural Network",
            "datasets": ["IMDB", "Penn Treebank"],
            "training": {
                "optimizer": "SGD",
                "loss": "BinaryCrossEntropy",
                "epochs": 100,
                "learning_rate": 0.01,
                "batch_size": 64
            },
            "architecture": {
                "layer_types": ["Embedding", "LSTM", "Dense", "Sigmoid"],
                "num_classes": 2,
                "hidden_dims": [128]
            }
        },
        {
            "algorithm_name": "Transformer Model",
            "datasets": ["WMT", "COCO"],
            "training": {
                "optimizer": "Adam",
                "loss": "CrossEntropyLoss",
                "epochs": 200,
                "learning_rate": 0.0001,
                "batch_size": 128
            },
            "architecture": {
                "layer_types": ["Embedding", "MultiHeadAttention", "LayerNorm", "Dense"],
                "num_classes": 50000,
                "hidden_dims": [512, 2048]
            }
        }
    ]

    papers = []
    for i in range(num_samples):
        template = templates[i % len(templates)]

        # Add some variation
        variant = json.loads(json.dumps(template))  # Deep copy
        variant["training"]["learning_rate"] *= (0.5 + i * 0.1)
        variant["training"]["epochs"] += i * 10

        paper = {
            "paper_id": f"synthetic_{i+1:03d}",
            "title": f"Synthetic Paper {i+1}",
            "text_file": f"synthetic_{i+1:03d}.txt",
            "ground_truth": variant
        }
        papers.append(paper)

    return papers


def generate_text_from_ground_truth(ground_truth: Dict) -> str:
    """Generate synthetic paper text from ground truth."""
    gt = ground_truth

    text = f"""
Title: {gt.get('algorithm_name', 'Deep Learning Model')}

Abstract:
This paper presents a novel {gt.get('algorithm_name', 'deep learning architecture')}
for solving complex machine learning tasks. Our approach achieves state-of-the-art
results on multiple benchmark datasets.

Methodology:
We propose {gt.get('algorithm_name', 'our method')} which uses a sophisticated
neural architecture to learn representations from data.

Architecture:
The model consists of multiple layers including {', '.join(gt.get('architecture', {}).get('layer_types', ['neural layers']))}.
The hidden dimensions are set to {gt.get('architecture', {}).get('hidden_dims', [128])}.
The output layer produces predictions for {gt.get('architecture', {}).get('num_classes', 10)} classes.

Training:
We train the model using the {gt.get('training', {}).get('optimizer', 'Adam')} optimizer
with an initial learning rate of {gt.get('training', {}).get('learning_rate', 0.001)}.
The batch size is set to {gt.get('training', {}).get('batch_size', 32)}.
We train for {gt.get('training', {}).get('epochs', 100)} epochs.
The loss function is {gt.get('training', {}).get('loss', 'cross-entropy')}.

Datasets:
We evaluate our method on {', '.join(gt.get('datasets', ['standard benchmarks']))}.

Results:
Our method achieves superior performance compared to baseline approaches.
"""
    return text


def main():
    parser = argparse.ArgumentParser(
        description="Generate calibration data for conformal prediction"
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("data/outputs"),
        help="Directory containing existing paper outputs"
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/calibration/validation_papers.json"),
        help="Output file for calibration data"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic calibration data"
    )
    parser.add_argument(
        "--num-synthetic",
        type=int,
        default=10,
        help="Number of synthetic samples to generate"
    )
    parser.add_argument(
        "--coverage-target",
        type=float,
        default=0.9,
        help="Target coverage level (default: 0.9 for 90%)"
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    papers = []

    # Extract from existing outputs
    if args.outputs_dir.exists() and not args.synthetic:
        logger.info(f"Extracting calibration data from {args.outputs_dir}")
        papers.extend(extract_from_existing_outputs(args.outputs_dir))

    # Generate synthetic data
    if args.synthetic or len(papers) < 5:
        logger.info(f"Generating {args.num_synthetic} synthetic calibration samples")
        papers.extend(create_synthetic_calibration_data(args.num_synthetic))

    if not papers:
        logger.error("No calibration data generated")
        return

    # Create output structure
    calibration_data = {
        "description": "Validation papers with ground truth annotations for conformal prediction calibration",
        "version": "1.0.0",
        "coverage_target": args.coverage_target,
        "papers": papers,
        "metadata": {
            "num_papers": len(papers),
            "fields_covered": [
                "algorithm_name",
                "datasets",
                "training.optimizer",
                "training.loss",
                "training.epochs",
                "training.learning_rate",
                "training.batch_size",
                "architecture.layer_types",
                "architecture.num_classes",
                "architecture.hidden_dims"
            ]
        }
    }

    # Write calibration file
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(calibration_data, indent=2))
    logger.info(f"Saved calibration data to {args.output_file}")

    # Generate text files for each paper
    calibration_dir = args.output_file.parent
    for paper in papers:
        text_file = calibration_dir / paper["text_file"]
        text = generate_text_from_ground_truth(paper["ground_truth"])
        text_file.write_text(text, encoding="utf-8")
        logger.info(f"Generated text file: {text_file}")

    logger.info(f"Calibration data generation complete: {len(papers)} papers")


if __name__ == "__main__":
    main()

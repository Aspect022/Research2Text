"""Pytest configuration and fixtures."""

import pytest
import sys
from pathlib import Path
import tempfile
import json

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_paper_text():
    """Sample paper text for testing."""
    return """
Title: CNN-BiLSTM with Attention for Facial Expression Recognition

Abstract:
This paper presents a novel deep learning architecture combining Convolutional Neural Networks (CNN),
Bidirectional Long Short-Term Memory (BiLSTM), and Attention Mechanisms for facial expression recognition.

Methodology:
We propose a hybrid model that leverages CNN for spatial feature extraction, BiLSTM for temporal modeling,
and attention mechanisms for focusing on relevant facial regions.

Architecture:
The model consists of the following layers:
- Conv2D layers for feature extraction
- BatchNorm for normalization
- ReLU activation
- MaxPool for downsampling
- BiLSTM for temporal modeling
- Attention mechanism for feature weighting
- Dense layers for classification
- Softmax for output

Training Configuration:
- Optimizer: Adam
- Loss: CrossEntropyLoss
- Epochs: 100
- Learning Rate: 0.001
- Batch Size: 32

Datasets:
We evaluate our method on FER2013, CK+, and JAFFE datasets.

Results:
Our method achieves 95% accuracy on FER2013, outperforming baseline methods.
"""


@pytest.fixture
def sample_method_struct():
    """Sample method structure for testing."""
    return {
        "algorithm_name": "CNN-BiLSTM with Attention",
        "paper_summary": "Hybrid model for facial expression recognition",
        "equations": ["h_t = LSTM(x_t, h_{t-1})"],
        "datasets": ["FER2013", "CK+", "JAFFE"],
        "training": {
            "optimizer": "Adam",
            "loss": "CrossEntropyLoss",
            "epochs": 100,
            "learning_rate": 0.001,
            "batch_size": 32
        },
        "architecture": {
            "layer_types": ["Conv2D", "BatchNorm", "ReLU", "MaxPool", "BiLSTM", "Attention", "Dense"],
            "num_classes": 7,
            "hidden_dims": [128, 256]
        }
    }


@pytest.fixture
def sample_sources():
    """Sample sources for validation testing."""
    return [
        {
            "id": "src_1",
            "title": "Attention Is All You Need",
            "venue": "NeurIPS",
            "year": 2017,
            "text": "We propose a new simple network architecture, the Transformer..."
        },
        {
            "id": "src_2",
            "title": "Deep Residual Learning",
            "venue": "CVPR",
            "year": 2016,
            "text": "We present a residual learning framework..."
        },
        {
            "id": "src_3",
            "title": "Random Blog Post",
            "venue": "Blog",
            "year": 2023,
            "text": "I think AI is cool and stuff..."
        }
    ]

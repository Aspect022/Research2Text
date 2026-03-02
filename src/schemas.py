"""
Pydantic schemas for the Research2Text pipeline.
Upgraded with confidence scoring for Phase 3 (Conformal Prediction).
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    optimizer: Optional[str] = Field(default=None)
    loss: Optional[str] = Field(default=None)
    epochs: Optional[int] = Field(default=None)
    learning_rate: Optional[float] = Field(default=None)
    batch_size: Optional[int] = Field(default=None)


class ConfidenceScore(BaseModel):
    """Confidence metadata for an extracted field.
    
    Implements a simplified conformal prediction framework:
    - `score` is the nonconformity score (0.0 = certain, 1.0 = uncertain)
    - `source` indicates where the value was extracted from
    - `evidence` stores the supporting text snippet from the paper
    """
    score: float = Field(
        default=0.5,
        ge=0.0, le=1.0,
        description="Confidence score: 0.0 = low confidence (likely hallucinated), 1.0 = high confidence (explicitly stated)"
    )
    source: str = Field(
        default="heuristic",
        description="Extraction source: 'explicit' (directly stated), 'inferred' (LLM inferred), 'heuristic' (regex-based), 'default' (template value)"
    )
    evidence: Optional[str] = Field(
        default=None,
        description="Supporting text snippet from the paper"
    )


class MethodStruct(BaseModel):
    algorithm_name: Optional[str] = Field(default=None)
    equations: List[str] = Field(default_factory=list)
    datasets: List[str] = Field(default_factory=list)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    inputs: Dict[str, str] = Field(default_factory=dict)
    outputs: Dict[str, str] = Field(default_factory=dict)
    references: List[str] = Field(default_factory=list)

    # Phase 3: Confidence scores for each extracted field
    confidence: Dict[str, ConfidenceScore] = Field(
        default_factory=dict,
        description="Per-field confidence scores from conformal prediction"
    )

    def overall_confidence(self) -> float:
        """Compute the mean confidence across all scored fields."""
        if not self.confidence:
            return 0.0
        scores = [c.score for c in self.confidence.values()]
        return sum(scores) / len(scores) if scores else 0.0

    def low_confidence_fields(self, threshold: float = 0.5) -> List[str]:
        """Return field names with confidence below the threshold."""
        return [
            field for field, conf in self.confidence.items()
            if conf.score < threshold
        ]


class GeneratedFile(BaseModel):
    path: str
    content: str


class RunResult(BaseModel):
    returncode: int
    stdout: str
    stderr: str


class ValidationResult(BaseModel):
    success: bool
    attempts: int
    last_error: Optional[str] = None
    logs_dir: Optional[str] = None
